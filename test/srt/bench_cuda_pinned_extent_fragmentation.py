import argparse
import json
import random
import time

import torch


def mib(value):
    return int(value * 1024 * 1024)


def sync():
    torch.cuda.synchronize()


def timed_copy(copy_fn, repeat, warmup):
    for _ in range(warmup):
        copy_fn()
    sync()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        copy_fn()
        sync()
        samples.append(time.perf_counter() - start)
    return samples


def summarize(name, direction, bytes_copied, copies_per_iter, samples):
    best = min(samples)
    avg = sum(samples) / len(samples)
    return {
        "name": name,
        "direction": direction,
        "bytes": bytes_copied,
        "copies_per_iter": copies_per_iter,
        "best_ms": best * 1000,
        "avg_ms": avg * 1000,
        "best_gib_s": bytes_copied / best / (1024**3),
        "avg_gib_s": bytes_copied / avg / (1024**3),
    }


def make_extents(total_bytes, extent_count):
    assert total_bytes % extent_count == 0
    extent_bytes = total_bytes // extent_count
    return [
        torch.empty(extent_bytes, dtype=torch.uint8, pin_memory=True)
        for _ in range(extent_count)
    ]


def chunk_ranges(total_bytes, chunk_bytes, shuffle):
    assert total_bytes % chunk_bytes == 0
    ranges = [(i, i + chunk_bytes) for i in range(0, total_bytes, chunk_bytes)]
    if shuffle:
        random.Random(0).shuffle(ranges)
    return ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-mib", type=int, default=512)
    parser.add_argument("--extent-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--page-kib", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)

    total_bytes = mib(args.total_mib)
    device = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    results = []

    for extent_count in args.extent_counts:
        if total_bytes % extent_count != 0:
            continue
        extents = make_extents(total_bytes, extent_count)
        extent_bytes = total_bytes // extent_count

        single_host = extents[0] if extent_count == 1 else torch.empty(
            total_bytes, dtype=torch.uint8, pin_memory=True
        )

        cases = []

        cases.append(
            (
                "single_contiguous",
                1,
                lambda: single_host.copy_(device, non_blocking=True),
                lambda: device.copy_(single_host, non_blocking=True),
            )
        )

        def d2h_extent_grouped():
            offset = 0
            for extent in extents:
                extent.copy_(device[offset : offset + extent_bytes], non_blocking=True)
                offset += extent_bytes

        def h2d_extent_grouped():
            offset = 0
            for extent in extents:
                device[offset : offset + extent_bytes].copy_(extent, non_blocking=True)
                offset += extent_bytes

        cases.append(
            (
                "extent_grouped",
                extent_count,
                d2h_extent_grouped,
                h2d_extent_grouped,
            )
        )

        for page_kib in args.page_kib:
            page_bytes = page_kib * 1024
            if total_bytes % page_bytes != 0 or extent_bytes % page_bytes != 0:
                continue
            ranges = chunk_ranges(total_bytes, page_bytes, shuffle=False)
            random_ranges = chunk_ranges(total_bytes, page_bytes, shuffle=True)

            def make_page_copy(ranges_to_copy):
                def d2h():
                    for start, end in ranges_to_copy:
                        extent_id = start // extent_bytes
                        local_start = start - extent_id * extent_bytes
                        local_end = local_start + (end - start)
                        extents[extent_id][local_start:local_end].copy_(
                            device[start:end], non_blocking=True
                        )

                def h2d():
                    for start, end in ranges_to_copy:
                        extent_id = start // extent_bytes
                        local_start = start - extent_id * extent_bytes
                        local_end = local_start + (end - start)
                        device[start:end].copy_(
                            extents[extent_id][local_start:local_end],
                            non_blocking=True,
                        )

                return d2h, h2d

            d2h_pages, h2d_pages = make_page_copy(ranges)
            cases.append(
                (
                    f"page_copy_{page_kib}kib",
                    len(ranges),
                    d2h_pages,
                    h2d_pages,
                )
            )

            d2h_random, h2d_random = make_page_copy(random_ranges)
            cases.append(
                (
                    f"random_page_copy_{page_kib}kib",
                    len(random_ranges),
                    d2h_random,
                    h2d_random,
                )
            )

        for name, copies_per_iter, d2h_fn, h2d_fn in cases:
            for direction, fn in (("D2H", d2h_fn), ("H2D", h2d_fn)):
                samples = timed_copy(fn, args.repeat, args.warmup)
                row = summarize(
                    name=name,
                    direction=direction,
                    bytes_copied=total_bytes,
                    copies_per_iter=copies_per_iter,
                    samples=samples,
                )
                row["total_mib"] = args.total_mib
                row["extent_count"] = extent_count
                row["extent_mib"] = extent_bytes / (1024 * 1024)
                results.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    print(json.dumps({"results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
