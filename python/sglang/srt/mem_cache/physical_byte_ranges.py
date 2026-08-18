"""Variable-size physical ranges for the Central I/O pinned arena."""

from __future__ import annotations


class FreeByteRanges:
    """A coalescing free list of half-open physical byte intervals.

    The arena itself is registered once.  This class only decides which
    physical bytes are currently unowned, so an interval released by one model
    can be aligned and reused by another model with a different KV geometry.
    """

    def __init__(self, ranges: list[tuple[int, int]] | None = None):
        self.ranges: list[tuple[int, int]] = []
        for start, end in ranges or []:
            if end <= start:
                raise ValueError("free byte range must have positive length")
            self.release(start, end - start)

    def allocate(self, byte_count: int, *, alignment: int) -> int | None:
        if byte_count <= 0:
            raise ValueError("allocated byte count must be positive")
        if alignment <= 0:
            raise ValueError("byte-range alignment must be positive")
        for index, (start, end) in enumerate(self.ranges):
            aligned_start = ((start + alignment - 1) // alignment) * alignment
            allocated_end = aligned_start + byte_count
            if allocated_end > end:
                continue
            replacement: list[tuple[int, int]] = []
            if start < aligned_start:
                replacement.append((start, aligned_start))
            if allocated_end < end:
                replacement.append((allocated_end, end))
            self.ranges[index : index + 1] = replacement
            return aligned_start
        return None

    def allocate_many(
        self, byte_count: int, *, alignment: int
    ) -> list[tuple[int, int]] | None:
        """Allocate a page-aligned request from one or more physical runs.

        The central arena can be physically fragmented after several live-KV
        handoffs.  A grow must therefore not fail merely because no single
        free run spans the full request.  Prefer the largest usable runs to
        minimize later DMA descriptors, but keep allocation atomic: failure
        restores every provisional run before returning ``None``.
        """
        if byte_count <= 0 or byte_count % alignment:
            raise ValueError("multi-run allocation must be positive and alignment sized")
        allocated: list[tuple[int, int]] = []
        remaining = byte_count
        try:
            while remaining:
                candidates: list[tuple[int, int]] = []
                for start, end in self.ranges:
                    aligned_start = ((start + alignment - 1) // alignment) * alignment
                    usable = ((end - aligned_start) // alignment) * alignment
                    if usable:
                        candidates.append((usable, aligned_start))
                if not candidates:
                    raise MemoryError
                usable, _ = max(candidates)
                take = min(remaining, usable)
                offset = self.allocate(take, alignment=alignment)
                if offset is None:
                    raise MemoryError
                allocated.append((offset, take))
                remaining -= take
        except MemoryError:
            for offset, count in allocated:
                self.release(offset, count)
            return None
        return allocated

    def release(self, start: int, byte_count: int) -> None:
        if start < 0 or byte_count <= 0:
            raise ValueError("released byte range must be non-negative and non-empty")
        end = start + byte_count
        merged: list[tuple[int, int]] = []
        inserted = False
        for current_start, current_end in self.ranges:
            if current_end < start:
                merged.append((current_start, current_end))
                continue
            if end < current_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((current_start, current_end))
                continue
            if current_start < end and start < current_end:
                raise ValueError("released byte range overlaps an existing free range")
            start = min(start, current_start)
            end = max(end, current_end)
        if not inserted:
            merged.append((start, end))
        self.ranges = merged
