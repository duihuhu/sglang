import argparse
import json
import time
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost


def bytes_to_mib(value):
    return value / (1024 * 1024)


def align_down(value, page_size):
    return (value // page_size) * page_size


def make_device_layers(layer_num, token_count, head_num, head_dim, dtype, device):
    k_layers = []
    v_layers = []
    token_ids = torch.arange(token_count, device=device, dtype=torch.float32)
    for layer_id in range(layer_num):
        k = torch.empty(
            (token_count, head_num, head_dim), dtype=dtype, device=device
        )
        v = torch.empty_like(k)
        page_codes = (token_ids // 64) % 1536
        k_scalar = (layer_id + 1) * 10 + page_codes
        v_scalar = (layer_id + 1) * 20 + page_codes
        k.copy_(k_scalar[:, None, None].to(dtype))
        v.copy_(v_scalar[:, None, None].to(dtype))
        k_layers.append(k)
        v_layers.append(v)
    return k_layers, v_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-tokens", type=int, default=8192)
    parser.add_argument("--host-ratio", type=float, default=2.0)
    parser.add_argument("--grow-tokens", type=int, default=4096)
    parser.add_argument("--transfer-tokens", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--layer-num", type=int, default=8)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    grow_tokens = align_down(args.grow_tokens, args.page_size)
    transfer_tokens = align_down(args.transfer_tokens, args.page_size)
    if grow_tokens <= 0 or transfer_tokens <= 0:
        raise ValueError("grow_tokens and transfer_tokens must be at least one page")

    device_pool_meta = SimpleNamespace(
        store_dtype=dtype,
        size=args.device_tokens,
        start_layer=0,
        end_layer=args.layer_num,
        layer_num=args.layer_num,
        head_num=args.head_num,
        head_dim=args.head_dim,
        device="cuda",
    )
    host_cache = MHATokenToKVPoolHost(
        device_pool=device_pool_meta,
        host_to_device_ratio=args.host_ratio,
        host_size=0,
        page_size=args.page_size,
        layout="page_first",
        pin_memory=True,
        device="cpu",
    )

    initial_tokens = host_cache.size
    initial_bytes = initial_tokens * host_cache.size_per_token
    extent_id = host_cache.grow_extent_online(grow_tokens)
    added_extent = host_cache.extent_table.extents[extent_id]
    added_bytes = added_extent.size * host_cache.size_per_token
    total_bytes = host_cache.size * host_cache.size_per_token

    old_transfer_tokens = min(transfer_tokens // 2, initial_tokens)
    old_transfer_tokens = align_down(old_transfer_tokens, args.page_size)
    new_transfer_tokens = transfer_tokens - old_transfer_tokens
    if new_transfer_tokens > added_extent.size:
        new_transfer_tokens = align_down(added_extent.size, args.page_size)
        old_transfer_tokens = transfer_tokens - new_transfer_tokens
    if old_transfer_tokens <= 0 or new_transfer_tokens <= 0:
        raise ValueError("Need both old and new extent transfer tokens")

    old_indices = host_cache.alloc(old_transfer_tokens)
    drain_old = host_cache.alloc(len(host_cache.extent_table.extents[0].free_slots))
    new_indices = host_cache.alloc(new_transfer_tokens)
    if old_indices is None or drain_old is None or new_indices is None:
        raise RuntimeError("Unexpected allocation failure")
    if not torch.all(old_indices < initial_tokens):
        raise RuntimeError("Old allocation did not come from the initial extent")
    if not torch.all(new_indices >= added_extent.base):
        raise RuntimeError("New allocation did not come from the appended extent")

    host_indices_cpu = torch.cat([old_indices, new_indices])
    device_indices = torch.arange(len(host_indices_cpu), dtype=torch.int64, device=device)
    host_indices = host_indices_cpu.to(device)

    source_k, source_v = make_device_layers(
        args.layer_num,
        len(host_indices_cpu),
        args.head_num,
        args.head_dim,
        dtype,
        device,
    )
    source_pool = SimpleNamespace(
        k_buffer=source_k,
        v_buffer=source_v,
        k_data_ptrs=torch.tensor(
            [x.data_ptr() for x in source_k], dtype=torch.uint64, device=device
        ),
        v_data_ptrs=torch.tensor(
            [x.data_ptr() for x in source_v], dtype=torch.uint64, device=device
        ),
    )

    torch.cuda.synchronize()
    backup_start = time.perf_counter()
    host_cache.backup_from_device_all_layer(
        source_pool, host_indices, device_indices, io_backend="kernel"
    )
    torch.cuda.synchronize()
    backup_ms = (time.perf_counter() - backup_start) * 1000

    restored_k = [torch.empty_like(x) for x in source_k]
    restored_v = [torch.empty_like(x) for x in source_v]
    restored_pool = SimpleNamespace(k_buffer=restored_k, v_buffer=restored_v)

    torch.cuda.synchronize()
    load_start = time.perf_counter()
    for layer_id in range(args.layer_num):
        host_cache.load_to_device_per_layer(
            restored_pool,
            host_indices,
            device_indices,
            layer_id=layer_id,
            io_backend="kernel",
        )
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000

    mismatches = []
    for layer_id in range(args.layer_num):
        if not torch.equal(restored_k[layer_id], source_k[layer_id]):
            mismatches.append(f"k{layer_id}")
        if not torch.equal(restored_v[layer_id], source_v[layer_id]):
            mismatches.append(f"v{layer_id}")

    result = {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
        "initial_tokens": initial_tokens,
        "grow_tokens": grow_tokens,
        "total_tokens_after_grow": host_cache.size,
        "old_transfer_tokens": old_transfer_tokens,
        "new_transfer_tokens": new_transfer_tokens,
        "size_per_token_bytes": host_cache.size_per_token,
        "initial_mib": bytes_to_mib(initial_bytes),
        "added_mib": bytes_to_mib(added_bytes),
        "total_mib_after_grow": bytes_to_mib(total_bytes),
        "backup_ms": backup_ms,
        "load_ms": load_ms,
        "old_extent_pinned": host_cache.extent_table.extents[0].kv_buffer.is_pinned(),
        "new_extent_pinned": added_extent.kv_buffer.is_pinned(),
    }
    print(json.dumps(result, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
