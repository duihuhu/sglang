import argparse
import json
import time
from types import SimpleNamespace

import torch
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_per_layer_direct_pf_lf,
)


def dtype_from_name(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


BENCH_STREAM = None


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


def make_extent_buffers(extent_count, pages_per_extent, layer_num, page_size, head_num, head_dim, dtype):
    return [
        torch.empty(
            (2, pages_per_extent, layer_num, page_size, head_num, head_dim),
            dtype=dtype,
            pin_memory=True,
        )
        for _ in range(extent_count)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-mib", type=int, default=512)
    parser.add_argument("--extent-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--layer-num", type=int, default=8)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
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

    src_k = [
        torch.empty((total_tokens, args.head_num, args.head_dim), dtype=dtype, device=device)
        for _ in range(args.layer_num)
    ]
    src_v = [torch.empty_like(x) for x in src_k]
    dst_k = [torch.empty_like(x) for x in src_k]
    dst_v = [torch.empty_like(x) for x in src_v]
    src_ptrs = src_k + src_v
    results = []
    for extent_count in args.extent_counts:
        if total_pages % extent_count != 0:
            continue
        pages_per_extent = total_pages // extent_count
        tokens_per_extent = pages_per_extent * args.page_size
        extents = make_extent_buffers(
            extent_count,
            pages_per_extent,
            args.layer_num,
            args.page_size,
            args.head_num,
            args.head_dim,
            dtype,
        )
        extent_indices = []
        token_offset = 0
        for _ in range(extent_count):
            extent_indices.append(
                SimpleNamespace(
                    local=torch.arange(tokens_per_extent, dtype=torch.int64),
                    global_=torch.arange(
                        token_offset,
                        token_offset + tokens_per_extent,
                        dtype=torch.int64,
                    ),
                )
            )
            token_offset += tokens_per_extent

        def d2h():
            for extent, indices in zip(extents, extent_indices):
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=src_ptrs,
                    dst_ptrs=[extent[0], extent[1]],
                    src_indices=indices.global_,
                    dst_indices=indices.local,
                    page_size=args.page_size,
                )

        def h2d():
            for extent, indices in zip(extents, extent_indices):
                for layer_id in range(args.layer_num):
                    transfer_kv_per_layer_direct_pf_lf(
                        src_ptrs=[extent[0], extent[1]],
                        dst_ptrs=[dst_k[layer_id], dst_v[layer_id]],
                        src_indices=indices.local,
                        dst_indices=indices.global_,
                        layer_id=layer_id,
                        page_size=args.page_size,
                    )

        for direction, fn in (("D2H", d2h), ("H2D", h2d)):
            best, avg = run_timed(fn, args.repeat, args.warmup)
            row = {
                "name": "sgl_kernel_direct_page_first",
                "direction": direction,
                "total_mib": total_bytes / (1024 * 1024),
                "extent_count": extent_count,
                "pages": total_pages,
                "cuda_batch_entries": total_pages * args.layer_num * 2,
                "api_calls_per_iter": extent_count
                if direction == "D2H"
                else extent_count * args.layer_num,
                "best_ms": best * 1000,
                "avg_ms": avg * 1000,
                "best_gib_s": total_bytes / best / (1024**3),
                "avg_gib_s": total_bytes / avg / (1024**3),
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    print(json.dumps({"results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
