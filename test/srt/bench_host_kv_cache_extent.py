import argparse
import json
import os
import tempfile
import time
from statistics import median
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import (
    HostKVCacheExtentTable,
    MHATokenToKVPoolHost,
)


def make_extent_indices(extent_count, pages_per_extent, page_size, sample_pages):
    extent_size = pages_per_extent * page_size
    starts = []
    for page_id in range(pages_per_extent):
        for extent_id in range(extent_count):
            starts.append(extent_id * extent_size + page_id * page_size)
            if len(starts) == sample_pages:
                break
        if len(starts) == sample_pages:
            break
    values = []
    for start in starts:
        values.extend(range(start, start + page_size))
    return torch.tensor(values, dtype=torch.int64), starts


def make_table(extent_count, pages_per_extent, page_size, layout, dtype, layer_num, head_num, head_dim):
    table = HostKVCacheExtentTable(page_size=page_size)
    extent_size = pages_per_extent * page_size
    for _ in range(extent_count):
        if layout == "page_first":
            kv_buffer = torch.empty(
                (2, extent_size, layer_num, head_num, head_dim),
                dtype=dtype,
                pin_memory=torch.cuda.is_available(),
            )
        elif layout == "page_head":
            kv_buffer = torch.empty(
                (2, pages_per_extent, head_num, page_size, layer_num, head_dim),
                dtype=dtype,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            raise ValueError(layout)
        kv_buffer.zero_()
        table.add_extent(num_slots=extent_size, kv_buffer=kv_buffer)
    return table


def make_host_cache(table, layout, page_size, dtype, layer_num, head_num, head_dim):
    host_cache = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
    host_cache.layout = layout
    host_cache.page_size = page_size
    host_cache.layer_num = layer_num
    host_cache.head_num = head_num
    host_cache.head_dim = head_dim
    host_cache.dtype = dtype
    host_cache.token_stride_size = head_num * head_dim * torch.tensor([], dtype=dtype).element_size()
    host_cache.layout_dim = host_cache.token_stride_size * layer_num
    host_cache.extent_table = table
    host_cache.device = "cpu"
    host_cache.pin_memory = torch.cuda.is_available()
    return host_cache


def make_device_pool(layer_num, device_size, head_num, head_dim, dtype, device):
    k_layers = []
    v_layers = []
    for layer_id in range(layer_num):
        k = torch.randn((device_size, head_num, head_dim), dtype=dtype, device=device)
        v = torch.randn_like(k)
        k += layer_id * 0.125
        v += layer_id * 0.25
        k_layers.append(k)
        v_layers.append(v)
    return SimpleNamespace(
        k_buffer=k_layers,
        v_buffer=v_layers,
        k_data_ptrs=torch.tensor([x.data_ptr() for x in k_layers], dtype=torch.uint64, device=device),
        v_data_ptrs=torch.tensor([x.data_ptr() for x in v_layers], dtype=torch.uint64, device=device),
    )


def cuda_event_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def wall_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return median(times)


def check_restored(restored_pool, source_pool, device_indices):
    for layer_id in range(len(source_pool.k_buffer)):
        if not torch.allclose(
            restored_pool.k_buffer[layer_id][device_indices],
            source_pool.k_buffer[layer_id][device_indices],
        ):
            return False
        if not torch.allclose(
            restored_pool.v_buffer[layer_id][device_indices],
            source_pool.v_buffer[layer_id][device_indices],
        ):
            return False
    return True


def benchmark_metadata(args):
    results = []
    for extent_count in args.extents:
        table = HostKVCacheExtentTable(page_size=args.page_size)
        for _ in range(extent_count):
            table.add_extent(args.pages_per_extent * args.page_size)
        host_indices, _ = make_extent_indices(
            extent_count, args.pages_per_extent, args.page_size, args.sample_pages
        )
        device_indices = torch.arange(len(host_indices), dtype=torch.int64)

        def group_once():
            groups = table.group_pages_by_extent(host_indices, device_indices)
            return len(groups)

        def descriptors_once():
            desc = table.page_descriptors(host_indices)
            return len(desc)

        for _ in range(args.warmup):
            group_once()
            descriptors_once()
        start = time.perf_counter()
        group_count = 0
        for _ in range(args.metadata_iters):
            group_count = group_once()
        group_ms = (time.perf_counter() - start) * 1000 / args.metadata_iters

        start = time.perf_counter()
        descriptor_count = 0
        for _ in range(args.metadata_iters):
            descriptor_count = descriptors_once()
        descriptor_ms = (time.perf_counter() - start) * 1000 / args.metadata_iters

        results.append(
            {
                "extent_count": extent_count,
                "tokens": int(len(host_indices)),
                "pages": int(len(host_indices) // args.page_size),
                "groups": group_count,
                "descriptors": descriptor_count,
                "group_ms": group_ms,
                "group_us_per_token": group_ms * 1000 / len(host_indices),
                "descriptor_ms": descriptor_ms,
                "descriptor_us_per_page": descriptor_ms * 1000 / descriptor_count,
            }
        )
    return results


def benchmark_transfer(args, layout):
    if not torch.cuda.is_available():
        return []
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32
    results = []

    for extent_count in args.extents:
        table = make_table(
            extent_count,
            args.pages_per_extent,
            args.page_size,
            layout,
            dtype,
            args.layer_num,
            args.head_num,
            args.head_dim,
        )
        host_cache = make_host_cache(
            table, layout, args.page_size, dtype, args.layer_num, args.head_num, args.head_dim
        )
        host_indices_cpu, _ = make_extent_indices(
            extent_count, args.pages_per_extent, args.page_size, args.sample_pages
        )
        token_count = len(host_indices_cpu)
        device_size = token_count * 2 + 1024
        device_indices = torch.randperm(device_size, device=device, dtype=torch.int64)[:token_count]
        host_indices = host_indices_cpu.to(device)
        source_pool = make_device_pool(
            args.layer_num, device_size, args.head_num, args.head_dim, dtype, device
        )
        restored_pool = make_device_pool(
            args.layer_num, device_size, args.head_num, args.head_dim, dtype, device
        )
        for layer in restored_pool.k_buffer + restored_pool.v_buffer:
            layer.fill_(float("nan"))

        def backup():
            host_cache.backup_from_device_all_layer(
                source_pool, host_indices, device_indices, io_backend="kernel"
            )

        def restore():
            for layer_id in range(args.layer_num):
                host_cache.load_to_device_per_layer(
                    restored_pool,
                    host_indices,
                    device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        def roundtrip():
            backup()
            restore()

        roundtrip()
        torch.cuda.synchronize()
        correct = check_restored(restored_pool, source_pool, device_indices)
        backup_ms = cuda_event_ms(backup, args.warmup, args.cuda_iters)
        restore_ms = cuda_event_ms(restore, args.warmup, args.cuda_iters)
        roundtrip_wall_ms = wall_ms(roundtrip, args.warmup, args.wall_iters)

        results.append(
            {
                "layout": layout,
                "extent_count": extent_count,
                "tokens": int(token_count),
                "pages": int(token_count // args.page_size),
                "groups": len(table.group_pages_by_extent(host_indices_cpu, torch.arange(token_count))),
                "correct": correct,
                "backup_event_ms": backup_ms,
                "restore_event_ms": restore_ms,
                "roundtrip_wall_ms": roundtrip_wall_ms,
                "event_ms_per_token": (backup_ms + restore_ms) / token_count,
            }
        )
    return results


def benchmark_compute_interference(args):
    if not torch.cuda.is_available():
        return []
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32
    results = []
    a = torch.randn((args.gemm_size, args.gemm_size), dtype=dtype, device=device)
    b = torch.randn_like(a)
    sink = torch.empty_like(a)

    def compute_once():
        sink.copy_(a @ b)

    compute_only_ms = wall_ms(compute_once, args.warmup, args.wall_iters)

    for extent_count in args.extents:
        table = make_table(
            extent_count,
            args.pages_per_extent,
            args.page_size,
            "page_first",
            dtype,
            args.layer_num,
            args.head_num,
            args.head_dim,
        )
        host_cache = make_host_cache(
            table, "page_first", args.page_size, dtype, args.layer_num, args.head_num, args.head_dim
        )
        host_indices_cpu, _ = make_extent_indices(
            extent_count, args.pages_per_extent, args.page_size, args.sample_pages
        )
        token_count = len(host_indices_cpu)
        device_size = token_count * 2 + 1024
        host_indices = host_indices_cpu.to(device)
        device_indices = torch.randperm(device_size, device=device, dtype=torch.int64)[:token_count]
        source_pool = make_device_pool(
            args.layer_num, device_size, args.head_num, args.head_dim, dtype, device
        )
        restored_pool = make_device_pool(
            args.layer_num, device_size, args.head_num, args.head_dim, dtype, device
        )

        def transfer_roundtrip():
            host_cache.backup_from_device_all_layer(
                source_pool, host_indices, device_indices, io_backend="kernel"
            )
            for layer_id in range(args.layer_num):
                host_cache.load_to_device_per_layer(
                    restored_pool,
                    host_indices,
                    device_indices,
                    layer_id=layer_id,
                    io_backend="kernel",
                )

        transfer_only_ms = wall_ms(transfer_roundtrip, args.warmup, args.wall_iters)

        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream = torch.cuda.Stream(device=device)

        def concurrent_once():
            with torch.cuda.stream(transfer_stream):
                transfer_roundtrip()
            with torch.cuda.stream(compute_stream):
                compute_once()

        concurrent_ms = wall_ms(concurrent_once, args.warmup, args.wall_iters)
        compute_after_ms = wall_ms(compute_once, args.warmup, args.wall_iters)

        results.append(
            {
                "extent_count": extent_count,
                "tokens": int(token_count),
                "compute_only_ms": compute_only_ms,
                "transfer_only_ms": transfer_only_ms,
                "concurrent_wall_ms": concurrent_ms,
                "max_of_isolated_ms": max(compute_only_ms, transfer_only_ms),
                "concurrent_over_max_ratio": concurrent_ms / max(compute_only_ms, transfer_only_ms),
                "compute_after_transfer_ms": compute_after_ms,
                "compute_after_over_baseline_ratio": compute_after_ms / compute_only_ms,
            }
        )
    return results


def benchmark_storage_marshalling(args):
    dtype = torch.float32
    results = []
    for extent_count in args.extents:
        table = make_table(
            extent_count,
            args.pages_per_extent,
            args.page_size,
            "page_first",
            dtype,
            args.layer_num,
            args.head_num,
            args.head_dim,
        )
        host_indices, page_starts = make_extent_indices(
            extent_count, args.pages_per_extent, args.page_size, args.sample_pages
        )
        page_starts = page_starts[: args.storage_pages]
        page_element_count = 2 * args.page_size * args.layer_num * args.head_num * args.head_dim
        for extent in table.extents:
            extent.kv_buffer.normal_()

        packed_pages = []

        def pack_once():
            packed_pages.clear()
            for start in page_starts:
                packed_pages.append(table.get_data_page(start, "page_first").clone())

        for _ in range(args.warmup):
            pack_once()
        start = time.perf_counter()
        for _ in range(args.metadata_iters):
            pack_once()
        pack_ms = (time.perf_counter() - start) * 1000 / args.metadata_iters

        fd, path = tempfile.mkstemp(prefix="extent_storage_probe_", suffix=".bin")
        os.close(fd)
        try:
            start = time.perf_counter()
            with open(path, "wb") as f:
                for page in packed_pages:
                    f.write(page.numpy().tobytes())
                f.flush()
                os.fsync(f.fileno())
            write_ms = (time.perf_counter() - start) * 1000

            restored_table = make_table(
                extent_count,
                args.pages_per_extent,
                args.page_size,
                "page_first",
                dtype,
                args.layer_num,
                args.head_num,
                args.head_dim,
            )
            start = time.perf_counter()
            with open(path, "rb") as f:
                for page_start in page_starts:
                    raw = f.read(page_element_count * torch.tensor([], dtype=dtype).element_size())
                    arr = np.frombuffer(raw, dtype=np.float32).copy()
                    restored_table.set_from_flat_data_page(
                        page_start, torch.from_numpy(arr), "page_first"
                    )
            read_restore_ms = (time.perf_counter() - start) * 1000
        finally:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        correct = True
        for page_start in page_starts:
            correct = correct and torch.equal(
                table.get_data_page(page_start, "page_first"),
                restored_table.get_data_page(page_start, "page_first"),
            )

        results.append(
            {
                "extent_count": extent_count,
                "pages": len(page_starts),
                "page_bytes": page_element_count * torch.tensor([], dtype=dtype).element_size(),
                "correct": correct,
                "pack_ms": pack_ms,
                "pack_us_per_page": pack_ms * 1000 / len(page_starts),
                "file_write_ms": write_ms,
                "file_read_restore_ms": read_restore_ms,
            }
        )
    return results


def make_file_backend(tmpdir):
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="extent-probe",
    )
    backend = HiCacheFile(config, file_path=tmpdir)
    backend.clear()
    return backend


def benchmark_hicache_file_backend(args):
    dtype = torch.float32
    results = []
    for extent_count in args.extents:
        source_table = make_table(
            extent_count,
            args.pages_per_extent,
            args.page_size,
            "page_first",
            dtype,
            args.layer_num,
            args.head_num,
            args.head_dim,
        )
        restored_table = make_table(
            extent_count,
            args.pages_per_extent,
            args.page_size,
            "page_first",
            dtype,
            args.layer_num,
            args.head_num,
            args.head_dim,
        )
        source_cache = make_host_cache(
            source_table,
            "page_first",
            args.page_size,
            dtype,
            args.layer_num,
            args.head_num,
            args.head_dim,
        )
        restored_cache = make_host_cache(
            restored_table,
            "page_first",
            args.page_size,
            dtype,
            args.layer_num,
            args.head_num,
            args.head_dim,
        )
        for extent_id, extent in enumerate(source_table.extents):
            extent.kv_buffer.normal_()
            extent.kv_buffer += extent_id * 10.0
        for extent in restored_table.extents:
            extent.kv_buffer.fill_(float("nan"))

        host_indices, page_starts = make_extent_indices(
            extent_count, args.pages_per_extent, args.page_size, args.sample_pages
        )
        page_starts = page_starts[: args.storage_pages]
        keys = [f"extent_{extent_count}_page_{i}" for i in range(len(page_starts))]

        with tempfile.TemporaryDirectory(prefix="hicache_file_extent_probe_") as tmpdir:
            backend = make_file_backend(tmpdir)

            def generic_page_set_once():
                data = [
                    source_cache.get_data_page(page_start)
                    for page_start in page_starts
                ]
                return backend.batch_set(keys, data)

            if not generic_page_set_once():
                raise RuntimeError("HiCacheFile batch_set failed during warmup")
            for key in keys:
                suffixed_key = backend._get_suffixed_key(key)
                path = os.path.join(tmpdir, f"{suffixed_key}.bin")
                os.remove(path)

            start = time.perf_counter()
            set_ok = generic_page_set_once()
            set_ms = (time.perf_counter() - start) * 1000

            def generic_page_get_once():
                dummy_pages = [
                    restored_cache.get_dummy_flat_data_page()
                    for _ in page_starts
                ]
                page_data = backend.batch_get(keys, dummy_pages)
                if page_data is None:
                    return False
                for i, data_page in enumerate(page_data):
                    if data_page is None:
                        return False
                    restored_cache.set_from_flat_data_page(page_starts[i], data_page)
                return True

            start = time.perf_counter()
            get_ok = generic_page_get_once()
            get_ms = (time.perf_counter() - start) * 1000

            file_bytes = 0
            for filename in os.listdir(tmpdir):
                path = os.path.join(tmpdir, filename)
                if os.path.isfile(path):
                    file_bytes += os.path.getsize(path)

        correct = set_ok and get_ok
        for page_start in page_starts:
            correct = correct and torch.equal(
                source_cache.get_data_page(page_start),
                restored_cache.get_data_page(page_start),
            )

        results.append(
            {
                "extent_count": extent_count,
                "pages": len(page_starts),
                "keys": len(keys),
                "correct": bool(correct),
                "batch_set_ms": set_ms,
                "batch_get_restore_ms": get_ms,
                "batch_set_us_per_page": set_ms * 1000 / len(page_starts),
                "batch_get_restore_us_per_page": get_ms * 1000 / len(page_starts),
                "file_bytes": file_bytes,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extents", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--pages-per-extent", type=int, default=64)
    parser.add_argument("--sample-pages", type=int, default=128)
    parser.add_argument("--storage-pages", type=int, default=64)
    parser.add_argument("--layer-num", type=int, default=4)
    parser.add_argument("--head-num", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--gemm-size", type=int, default=2048)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--metadata-iters", type=int, default=50)
    parser.add_argument("--cuda-iters", type=int, default=20)
    parser.add_argument("--wall-iters", type=int, default=10)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    result = {
        "config": vars(args),
        "metadata": benchmark_metadata(args),
        "storage_marshalling": benchmark_storage_marshalling(args),
        "hicache_file_backend": benchmark_hicache_file_backend(args),
        "transfer_page_first": benchmark_transfer(args, "page_first"),
        "transfer_page_head": benchmark_transfer(args, "page_head"),
        "compute_interference": benchmark_compute_interference(args),
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
