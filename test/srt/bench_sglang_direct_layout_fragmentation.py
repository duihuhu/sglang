import argparse
import json
import time
from types import SimpleNamespace

import torch
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_direct,
    transfer_kv_per_layer_direct_pf_lf,
)


BENCH_STREAM = None


def dtype_from_name(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def sync():
    torch.cuda.synchronize()


def run_timed(fn, repeat, warmup):
    global BENCH_STREAM
    if BENCH_STREAM is None:
        BENCH_STREAM = torch.cuda.Stream()

    def run_on_bench_stream():
        BENCH_STREAM.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(BENCH_STREAM):
            fn()
        torch.cuda.current_stream().wait_stream(BENCH_STREAM)

    for _ in range(warmup):
        run_on_bench_stream()
    sync()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        run_on_bench_stream()
        sync()
        samples.append(time.perf_counter() - start)
    return min(samples), sum(samples) / len(samples)


def make_page_pattern(total_tokens, page_size, pattern, seed):
    total_pages = total_tokens // page_size
    if pattern == "contiguous":
        pages = torch.arange(total_pages, dtype=torch.int64)
    elif pattern == "page_reverse":
        pages = torch.arange(total_pages - 1, -1, -1, dtype=torch.int64)
    elif pattern == "page_shuffle":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        pages = torch.randperm(total_pages, generator=generator, dtype=torch.int64)
    else:
        raise ValueError(f"Unsupported index pattern: {pattern}")

    offsets = torch.arange(page_size, dtype=torch.int64)
    return (pages[:, None] * page_size + offsets[None, :]).reshape(-1)


def make_extent_local_pattern(tokens_per_extent, page_size, pattern, seed, extent_id):
    if pattern not in ["host_page_shuffle", "both_page_shuffle"]:
        return torch.arange(tokens_per_extent, dtype=torch.int64)

    pages_per_extent = tokens_per_extent // page_size
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009 * (extent_id + 1))
    pages = torch.randperm(pages_per_extent, generator=generator, dtype=torch.int64)
    offsets = torch.arange(page_size, dtype=torch.int64)
    return (pages[:, None] * page_size + offsets[None, :]).reshape(-1)


def get_page_run_stats(indices, page_size):
    page_ids = [
        int(indices[offset].item()) // page_size
        for offset in range(0, len(indices), page_size)
    ]
    if not page_ids:
        return {"page_runs": 0, "max_run_pages": 0, "avg_run_pages": 0.0}
    runs = []
    current_run = 1
    for previous, current in zip(page_ids, page_ids[1:]):
        if current == previous + 1:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    return {
        "page_runs": len(runs),
        "max_run_pages": max(runs),
        "avg_run_pages": float(sum(runs) / len(runs)),
    }


def split_indices_by_extent(global_indices, extent_count, page_size, pattern, seed):
    assert len(global_indices) % extent_count == 0
    tokens_per_extent = len(global_indices) // extent_count
    assert tokens_per_extent % page_size == 0

    ret = []
    for extent_id in range(extent_count):
        start = extent_id * tokens_per_extent
        end = start + tokens_per_extent
        local_indices = make_extent_local_pattern(
            tokens_per_extent, page_size, pattern, seed, extent_id
        )
        ret.append(
            SimpleNamespace(
                local=local_indices,
                global_=global_indices[start:end].contiguous(),
            )
        )
    return ret


def make_device_layers(total_tokens, layer_num, head_num, head_dim, dtype, device):
    k_layers = [
        torch.empty((total_tokens, head_num, head_dim), dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    v_layers = [torch.empty_like(k) for k in k_layers]
    return k_layers, v_layers


def make_layer_first_extents(
    extent_count, tokens_per_extent, layer_num, head_num, head_dim, dtype
):
    extents = []
    for _ in range(extent_count):
        k_layers = [
            torch.empty(
                (tokens_per_extent, head_num, head_dim),
                dtype=dtype,
                pin_memory=True,
            )
            for _ in range(layer_num)
        ]
        v_layers = [
            torch.empty(
                (tokens_per_extent, head_num, head_dim),
                dtype=dtype,
                pin_memory=True,
            )
            for _ in range(layer_num)
        ]
        extents.append(SimpleNamespace(k=k_layers, v=v_layers))
    return extents


def make_page_first_direct_extents(
    extent_count, pages_per_extent, layer_num, page_size, head_num, head_dim, dtype
):
    return [
        torch.empty(
            (2, pages_per_extent, layer_num, page_size, head_num, head_dim),
            dtype=dtype,
            pin_memory=True,
        )
        for _ in range(extent_count)
    ]


def build_layer_first_direct_case(
    src_k, src_v, dst_k, dst_v, extent_indices, layer_first_extents, page_size
):
    src_ptrs = src_k + src_v

    def d2h():
        for extent, indices in zip(layer_first_extents, extent_indices):
            transfer_kv_direct(
                src_layers=src_ptrs,
                dst_layers=extent.k + extent.v,
                src_indices=indices.global_,
                dst_indices=indices.local,
                page_size=page_size,
            )

    def h2d():
        for extent, indices in zip(layer_first_extents, extent_indices):
            for layer_id in range(len(src_k)):
                transfer_kv_direct(
                    src_layers=[extent.k[layer_id], extent.v[layer_id]],
                    dst_layers=[dst_k[layer_id], dst_v[layer_id]],
                    src_indices=indices.local,
                    dst_indices=indices.global_,
                    page_size=page_size,
                )

    return d2h, h2d


def build_page_first_direct_case(
    src_k, src_v, dst_k, dst_v, extent_indices, page_first_extents, page_size
):
    src_ptrs = src_k + src_v

    def d2h():
        for extent, indices in zip(page_first_extents, extent_indices):
            transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=src_ptrs,
                dst_ptrs=[extent[0], extent[1]],
                src_indices=indices.global_,
                dst_indices=indices.local,
                page_size=page_size,
            )

    def h2d():
        for extent, indices in zip(page_first_extents, extent_indices):
            for layer_id in range(len(src_k)):
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[extent[0], extent[1]],
                    dst_ptrs=[dst_k[layer_id], dst_v[layer_id]],
                    src_indices=indices.local,
                    dst_indices=indices.global_,
                    layer_id=layer_id,
                    page_size=page_size,
                )

    return d2h, h2d


def build_naive_page_loop_case(
    src_k, src_v, dst_k, dst_v, extent_indices, layer_first_extents, page_size
):
    layer_num = len(src_k)

    def iter_page_pairs(indices):
        for offset in range(0, len(indices.local), page_size):
            local_start = int(indices.local[offset].item())
            global_start = int(indices.global_[offset].item())
            yield local_start, global_start

    def d2h():
        for extent, indices in zip(layer_first_extents, extent_indices):
            for local_start, global_start in iter_page_pairs(indices):
                local_slice = slice(local_start, local_start + page_size)
                global_slice = slice(global_start, global_start + page_size)
                for layer_id in range(layer_num):
                    extent.k[layer_id][local_slice].copy_(
                        src_k[layer_id][global_slice], non_blocking=True
                    )
                    extent.v[layer_id][local_slice].copy_(
                        src_v[layer_id][global_slice], non_blocking=True
                    )

    def h2d():
        for extent, indices in zip(layer_first_extents, extent_indices):
            for local_start, global_start in iter_page_pairs(indices):
                local_slice = slice(local_start, local_start + page_size)
                global_slice = slice(global_start, global_start + page_size)
                for layer_id in range(layer_num):
                    dst_k[layer_id][global_slice].copy_(
                        extent.k[layer_id][local_slice], non_blocking=True
                    )
                    dst_v[layer_id][global_slice].copy_(
                        extent.v[layer_id][local_slice], non_blocking=True
                    )

    return d2h, h2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-mib", type=int, default=512)
    parser.add_argument("--extent-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["layer_first_direct", "page_first_direct", "naive_page_loop"],
        default=["layer_first_direct", "page_first_direct", "naive_page_loop"],
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=[
            "contiguous",
            "page_reverse",
            "page_shuffle",
            "host_page_shuffle",
            "both_page_shuffle",
        ],
        default=["contiguous", "page_shuffle"],
    )
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--layer-num", type=int, default=8)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = dtype_from_name(args.dtype)
    elem_size = torch.empty((), dtype=dtype).element_size()
    bytes_per_token = args.layer_num * args.head_num * args.head_dim * elem_size * 2
    total_tokens = (args.total_mib * 1024 * 1024) // bytes_per_token
    total_tokens = (total_tokens // args.page_size) * args.page_size
    total_bytes = total_tokens * bytes_per_token
    total_pages = total_tokens // args.page_size

    src_k, src_v = make_device_layers(
        total_tokens, args.layer_num, args.head_num, args.head_dim, dtype, device
    )
    dst_k, dst_v = make_device_layers(
        total_tokens, args.layer_num, args.head_num, args.head_dim, dtype, device
    )

    results = []
    for pattern in args.patterns:
        global_pattern = "page_shuffle" if pattern == "both_page_shuffle" else pattern
        if pattern == "host_page_shuffle":
            global_pattern = "contiguous"
        global_indices = make_page_pattern(
            total_tokens, args.page_size, global_pattern, args.seed
        )
        for extent_count in args.extent_counts:
            if total_pages % extent_count != 0:
                continue
            pages_per_extent = total_pages // extent_count
            tokens_per_extent = pages_per_extent * args.page_size
            extent_indices = split_indices_by_extent(
                global_indices, extent_count, args.page_size, pattern, args.seed
            )
            local_stats = [
                get_page_run_stats(indices.local, args.page_size)
                for indices in extent_indices
            ]
            local_page_runs = sum(stats["page_runs"] for stats in local_stats)
            local_max_run_pages = max(
                (stats["max_run_pages"] for stats in local_stats), default=0
            )
            local_avg_run_pages = (
                float(total_pages / local_page_runs) if local_page_runs else 0.0
            )

            layer_first_extents = None
            page_first_extents = None
            for layout in args.layouts:
                if layout in ["layer_first_direct", "naive_page_loop"]:
                    if layer_first_extents is None:
                        layer_first_extents = make_layer_first_extents(
                            extent_count,
                            tokens_per_extent,
                            args.layer_num,
                            args.head_num,
                            args.head_dim,
                            dtype,
                        )
                    d2h, h2d = (
                        build_layer_first_direct_case(
                            src_k,
                            src_v,
                            dst_k,
                            dst_v,
                            extent_indices,
                            layer_first_extents,
                            args.page_size,
                        )
                        if layout == "layer_first_direct"
                        else build_naive_page_loop_case(
                            src_k,
                            src_v,
                            dst_k,
                            dst_v,
                            extent_indices,
                            layer_first_extents,
                            args.page_size,
                        )
                    )
                    api_calls_d2h = (
                        extent_count
                        if layout == "layer_first_direct"
                        else extent_count * pages_per_extent * args.layer_num * 2
                    )
                    api_calls_h2d = (
                        extent_count * args.layer_num
                        if layout == "layer_first_direct"
                        else extent_count * pages_per_extent * args.layer_num * 2
                    )
                elif layout == "page_first_direct":
                    if page_first_extents is None:
                        page_first_extents = make_page_first_direct_extents(
                            extent_count,
                            pages_per_extent,
                            args.layer_num,
                            args.page_size,
                            args.head_num,
                            args.head_dim,
                            dtype,
                        )
                    d2h, h2d = build_page_first_direct_case(
                        src_k,
                        src_v,
                        dst_k,
                        dst_v,
                        extent_indices,
                        page_first_extents,
                        args.page_size,
                    )
                    api_calls_d2h = extent_count
                    api_calls_h2d = extent_count * args.layer_num
                else:
                    raise ValueError(f"Unsupported layout: {layout}")

                for direction, fn, api_calls in (
                    ("D2H", d2h, api_calls_d2h),
                    ("H2D", h2d, api_calls_h2d),
                ):
                    best, avg = run_timed(fn, args.repeat, args.warmup)
                    row = {
                        "name": "sgl_direct_layout_fragmentation",
                        "layout": layout,
                        "pattern": pattern,
                        "direction": direction,
                        "total_mib": total_bytes / (1024 * 1024),
                        "extent_count": extent_count,
                        "extent_mib": total_bytes / extent_count / (1024 * 1024),
                        "pages": total_pages,
                        "pages_per_extent": pages_per_extent,
                        "host_page_runs": local_page_runs,
                        "host_max_run_pages": local_max_run_pages,
                        "host_avg_run_pages": local_avg_run_pages,
                        "api_calls_per_iter": api_calls,
                        "best_ms": best * 1000,
                        "avg_ms": avg * 1000,
                        "best_gib_s": total_bytes / best / (1024**3),
                        "avg_gib_s": total_bytes / avg / (1024**3),
                    }
                    results.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    print(json.dumps({"results": results}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
