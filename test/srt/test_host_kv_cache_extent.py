import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool_host import (
    HostKVCacheExtentGroup,
    HostKVCacheExtentTable,
    HostKVCache,
    MLATokenToKVPoolHost,
    MHATokenToKVPoolHost,
)


class TestHostKVCacheExtentTable(unittest.TestCase):
    def _new_runtime_mha_host_cache(self, layout="page_first", pin_memory=False):
        device_pool = SimpleNamespace(
            store_dtype=torch.float32,
            size=1024,
            start_layer=0,
            end_layer=2,
            layer_num=2,
            head_num=1,
            head_dim=64,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return MHATokenToKVPoolHost(
            device_pool=device_pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=64,
            layout=layout,
            pin_memory=pin_memory,
            device="cpu",
        )

    def test_runtime_online_grow_adds_pinned_extent_without_losing_existing_pages(self):
        host_cache = self._new_runtime_mha_host_cache(
            "page_first", pin_memory=torch.cuda.is_available()
        )
        old_size = host_cache.size
        old_available = host_cache.available_size()
        old_indices = host_cache.alloc(host_cache.page_size)
        old_page_start = int(old_indices[0])
        old_page = 9000 + torch.arange(
            2 * host_cache.layer_num * host_cache.page_size * host_cache.head_num * host_cache.head_dim,
            dtype=host_cache.dtype,
        )
        host_cache.set_from_flat_data_page(old_page_start, old_page)

        new_extent_id = host_cache.grow_extent_online(host_cache.page_size * 2)

        self.assertEqual(new_extent_id, 1)
        self.assertEqual(host_cache.size, old_size + host_cache.page_size * 2)
        self.assertEqual(
            host_cache.available_size(),
            old_available - host_cache.page_size + host_cache.page_size * 2,
        )
        self.assertTrue(hasattr(host_cache, "extent_table"))
        self.assertEqual(len(host_cache.extent_table.extents), 2)
        if torch.cuda.is_available():
            self.assertTrue(host_cache.extent_table.extents[1].kv_buffer.is_pinned())

        restored_old_page = host_cache.get_data_page(old_page_start)
        self.assertTrue(torch.equal(restored_old_page, old_page))

        remaining_old_free = host_cache.alloc(old_available - host_cache.page_size)
        self.assertTrue(torch.all(remaining_old_free < old_size))

        new_indices = host_cache.alloc(host_cache.page_size * 2)
        self.assertTrue(torch.all(new_indices >= old_size))
        new_page = 12000 + torch.arange(
            2 * host_cache.layer_num * host_cache.page_size * host_cache.head_num * host_cache.head_dim,
            dtype=host_cache.dtype,
        )
        host_cache.set_from_flat_data_page(int(new_indices[0]), new_page)
        self.assertTrue(torch.equal(host_cache.get_data_page(int(new_indices[0])), new_page))

    def test_extent_table_clear_preserves_draining_state(self):
        table = HostKVCacheExtentTable(page_size=4)
        table.add_extent(num_slots=8)
        table.add_extent(num_slots=8)
        table.mark_draining(0)

        table.clear()

        self.assertEqual(table.extent_state(0), HostKVCacheExtentTable.DRAINING)
        self.assertEqual(table.extent_state(1), HostKVCacheExtentTable.ACTIVE)
        self.assertEqual(table.available_size(), 8)
        allocated = table.alloc(8)
        self.assertTrue(torch.all(allocated >= 8))
        self.assertIsNone(table.alloc(4))

    def test_reserve_from_extent_keeps_extent_active_but_consumes_free_pages(self):
        table = HostKVCacheExtentTable(page_size=4)
        table.add_extent(num_slots=12)
        table.add_extent(num_slots=8)

        reserved = table.reserve_from_extent(extent_id=0, num_slots=8)

        self.assertEqual(table.extent_state(0), HostKVCacheExtentTable.ACTIVE)
        self.assertEqual(table.extent_state(1), HostKVCacheExtentTable.ACTIVE)
        self.assertEqual(reserved.tolist(), list(range(8)))
        self.assertEqual(len(table.extents[0].free_slots), 4)
        self.assertEqual(table.available_size(), 12)

        allocated = table.alloc(12)
        self.assertEqual(allocated.tolist(), list(range(8, 20)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for runtime transfer grow test")
    def test_runtime_online_grow_new_extent_works_with_production_transfer(self):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        host_cache = self._new_runtime_mha_host_cache("page_first", pin_memory=True)
        host_cache.grow_extent_online(host_cache.page_size * 2)

        old_free_count = len(host_cache.extent_table.extents[0].free_slots)
        old_indices = host_cache.alloc(old_free_count)
        self.assertTrue(torch.all(old_indices < host_cache.extent_table.extents[1].base))

        new_indices_cpu = host_cache.alloc(host_cache.page_size * 2)
        self.assertTrue(torch.all(new_indices_cpu >= host_cache.extent_table.extents[1].base))
        host_indices = new_indices_cpu.to(device)
        device_indices = torch.arange(len(new_indices_cpu), dtype=torch.int64, device=device)

        dev_k_layers = []
        dev_v_layers = []
        for layer_id in range(host_cache.layer_num):
            k = torch.zeros(
                (len(new_indices_cpu) + 16, host_cache.head_num, host_cache.head_dim),
                dtype=host_cache.dtype,
                device=device,
            )
            v = torch.zeros_like(k)
            for idx in range(len(new_indices_cpu) + 16):
                k[idx, 0, :] = 1000 * (layer_id + 1) + idx * 10 + torch.arange(
                    host_cache.head_dim, device=device
                )
                v[idx, 0, :] = 2000 * (layer_id + 1) + idx * 10 + torch.arange(
                    host_cache.head_dim, device=device
                )
            dev_k_layers.append(k)
            dev_v_layers.append(v)

        source_pool = SimpleNamespace(
            k_buffer=dev_k_layers,
            v_buffer=dev_v_layers,
            k_data_ptrs=torch.tensor(
                [x.data_ptr() for x in dev_k_layers], dtype=torch.uint64, device=device
            ),
            v_data_ptrs=torch.tensor(
                [x.data_ptr() for x in dev_v_layers], dtype=torch.uint64, device=device
            ),
        )
        host_cache.backup_from_device_all_layer(
            source_pool, host_indices, device_indices, io_backend="kernel"
        )
        torch.cuda.synchronize()

        restored_k_layers = [torch.full_like(x, -1) for x in dev_k_layers]
        restored_v_layers = [torch.full_like(x, -1) for x in dev_v_layers]
        restored_pool = SimpleNamespace(
            k_buffer=restored_k_layers,
            v_buffer=restored_v_layers,
        )
        for layer_id in range(host_cache.layer_num):
            host_cache.load_to_device_per_layer(
                restored_pool,
                host_indices,
                device_indices,
                layer_id=layer_id,
                io_backend="kernel",
            )
        torch.cuda.synchronize()

        for layer_id in range(host_cache.layer_num):
            self.assertTrue(
                torch.equal(
                    restored_k_layers[layer_id][device_indices],
                    dev_k_layers[layer_id][device_indices],
                )
            )
            self.assertTrue(
                torch.equal(
                    restored_v_layers[layer_id][device_indices],
                    dev_v_layers[layer_id][device_indices],
                )
            )

    def _new_mha_host_cache_with_extents(
        self, layout, extent0_buffer, extent1_buffer, page_size=4
    ):
        host_cache = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host_cache.layout = layout
        host_cache.page_size = page_size
        host_cache.layer_num = 1
        host_cache.head_num = 1
        host_cache.head_dim = 1
        host_cache.dtype = extent0_buffer.dtype

        table = HostKVCacheExtentTable(page_size=page_size)
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)
        host_cache.extent_table = table
        return host_cache

    def _new_mla_host_cache_with_extents(
        self, layout, extent0_buffer, extent1_buffer, page_size=4
    ):
        host_cache = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host_cache.layout = layout
        host_cache.page_size = page_size
        host_cache.layer_num = 2
        host_cache.kv_cache_dim = 3
        host_cache.dtype = extent0_buffer.dtype

        table = HostKVCacheExtentTable(page_size=page_size)
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)
        host_cache.extent_table = table
        return host_cache

    def test_mla_get_and_set_data_page_can_use_extent_table_without_single_kv_buffer(self):
        extent0_buffer = torch.zeros((8, 2, 1, 3), dtype=torch.int64)
        extent1_buffer = torch.zeros_like(extent0_buffer)
        host_cache = self._new_mla_host_cache_with_extents(
            "page_first", extent0_buffer, extent1_buffer
        )

        data_page = 4000 + torch.arange(4 * 2 * 1 * 3, dtype=torch.int64)
        host_cache.set_from_flat_data_page(8, data_page)
        restored = host_cache.get_data_page(8)

        self.assertTrue(torch.equal(restored, data_page))
        self.assertTrue(torch.equal(extent1_buffer[0:4, :, :, :].flatten(), data_page))
        self.assertTrue(torch.equal(extent0_buffer, torch.zeros_like(extent0_buffer)))

    def test_mla_page_buffer_meta_can_use_extent_table_without_single_kv_buffer(self):
        extent0_buffer = torch.empty((8, 2, 1, 3), dtype=torch.float32)
        extent1_buffer = torch.empty_like(extent0_buffer)
        host_cache = self._new_mla_host_cache_with_extents(
            "page_first", extent0_buffer, extent1_buffer
        )

        ptrs, sizes = host_cache.get_page_buffer_meta(torch.tensor([8, 9, 10, 11]))

        item_size = extent1_buffer.element_size()
        self.assertEqual(ptrs, [extent1_buffer.data_ptr()])
        self.assertEqual(sizes, [2 * item_size * 4 * 3])

    def test_mha_get_and_set_data_page_can_use_extent_table_without_single_kv_buffer(self):
        extent0_buffer = torch.zeros((2, 8, 1, 1, 1), dtype=torch.int64)
        extent1_buffer = torch.zeros((2, 8, 1, 1, 1), dtype=torch.int64)
        host_cache = self._new_mha_host_cache_with_extents(
            "page_first", extent0_buffer, extent1_buffer
        )

        data_page = 3000 + torch.arange(2 * 4 * 1 * 1 * 1, dtype=torch.int64)
        host_cache.set_from_flat_data_page(8, data_page)
        restored = host_cache.get_data_page(8)

        self.assertTrue(torch.equal(restored, data_page))
        self.assertTrue(torch.equal(extent1_buffer[:, 0:4, :, :, :].flatten(), data_page))
        self.assertTrue(torch.equal(extent0_buffer, torch.zeros_like(extent0_buffer)))

    def test_mha_page_buffer_meta_can_use_extent_table_without_single_kv_buffer(self):
        extent0_buffer = torch.empty((2, 8, 1, 1, 1), dtype=torch.float32)
        extent1_buffer = torch.empty_like(extent0_buffer)
        host_cache = self._new_mha_host_cache_with_extents(
            "page_first", extent0_buffer, extent1_buffer
        )

        ptrs, sizes = host_cache.get_page_buffer_meta(torch.tensor([8, 9, 10, 11]))

        item_size = extent1_buffer.element_size()
        v_offset = 1 * 8 * 1 * 1 * item_size
        self.assertEqual(ptrs, [extent1_buffer.data_ptr(), extent1_buffer.data_ptr() + v_offset])
        self.assertEqual(sizes, [1 * item_size * 4 * 1 * 1, 1 * item_size * 4 * 1 * 1])

    def test_mha_split_heads_page_buffer_meta_can_use_extent_table_without_single_kv_buffer(self):
        extent0_buffer = torch.empty((2, 2, 4, 4, 2, 2), dtype=torch.float32)
        extent1_buffer = torch.empty_like(extent0_buffer)
        host_cache = self._new_mha_host_cache_with_extents(
            "page_head", extent0_buffer, extent1_buffer
        )
        host_cache.layer_num = 2
        host_cache.head_num = 4
        host_cache.head_dim = 2

        ptrs, sizes = host_cache.get_split_heads_page_buffer_meta(
            torch.tensor([8, 9, 10, 11]), split_factor=2
        )

        item_size = extent1_buffer.element_size()
        v_offset = 2 * 8 * 4 * 2 * item_size
        head_stride = 4 * 2 * 2 * item_size
        self.assertEqual(
            ptrs,
            [
                extent1_buffer.data_ptr(),
                extent1_buffer.data_ptr() + v_offset,
                extent1_buffer.data_ptr() + 2 * head_stride,
                extent1_buffer.data_ptr() + 2 * head_stride + v_offset,
            ],
        )
        self.assertEqual(sizes, [2 * item_size * 4 * 4 * 2 // 2] * 4)

    def test_grow_keeps_old_indices_and_allocates_from_new_extent(self):
        table = HostKVCacheExtentTable(page_size=4)

        first = table.add_extent(num_slots=8)
        old_indices = table.alloc(4)

        self.assertEqual(first, 0)
        self.assertTrue(torch.equal(old_indices, torch.tensor([0, 1, 2, 3])))
        self.assertEqual(table.resolve_index(0), (0, 0))
        self.assertEqual(table.resolve_index(3), (0, 3))

        second = table.add_extent(num_slots=8)
        new_indices = table.alloc(8)

        self.assertEqual(second, 1)
        self.assertTrue(torch.equal(new_indices, torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])))
        self.assertEqual(table.resolve_index(4), (0, 4))
        self.assertEqual(table.resolve_index(8), (1, 0))

        # Old radix-tree host_value-style indices still point to the original extent.
        self.assertEqual(table.resolve_indices(old_indices), [(0, 0), (0, 1), (0, 2), (0, 3)])

    def test_draining_extent_stops_new_allocations_but_keeps_old_indices_valid(self):
        table = HostKVCacheExtentTable(page_size=4)
        table.add_extent(num_slots=8)
        table.add_extent(num_slots=8)

        old_indices = table.alloc(8)
        table.mark_draining(0)
        new_indices = table.alloc(4)

        self.assertTrue(torch.equal(old_indices, torch.arange(0, 8)))
        self.assertTrue(torch.equal(new_indices, torch.tensor([8, 9, 10, 11])))
        self.assertEqual(table.resolve_index(2), (0, 2))
        self.assertEqual(table.extent_state(0), "draining")

    def test_free_returns_slots_to_their_original_extent(self):
        table = HostKVCacheExtentTable(page_size=4)
        table.add_extent(num_slots=8)
        table.add_extent(num_slots=8)

        allocated = table.alloc(12)
        table.free(allocated[:4])
        reused = table.alloc(4)

        self.assertTrue(torch.equal(reused, torch.tensor([0, 1, 2, 3])))
        self.assertEqual(table.resolve_index(int(reused[-1])), (0, 3))

    def test_groups_transfer_indices_by_extent_without_losing_device_pairing(self):
        table = HostKVCacheExtentTable(page_size=4)
        table.add_extent(num_slots=8)
        table.add_extent(num_slots=8)

        host_indices = torch.tensor([2, 8, 3, 9, 10, 4])
        device_indices = torch.tensor([20, 21, 22, 23, 24, 25])

        groups = table.group_indices_by_extent(host_indices, paired_indices=device_indices)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].extent_id, 0)
        self.assertTrue(torch.equal(groups[0].local_indices, torch.tensor([2, 3, 4])))
        self.assertTrue(torch.equal(groups[0].paired_indices, torch.tensor([20, 22, 25])))
        self.assertEqual(groups[1].extent_id, 1)
        self.assertTrue(torch.equal(groups[1].local_indices, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(groups[1].paired_indices, torch.tensor([21, 23, 24])))

    def test_extent_transfer_trace_records_page_run_fragmentation(self):
        class TraceOnlyHostKVCache(HostKVCache):
            def get_size_per_token(self):
                return 0

            def init_kv_buffer(self):
                return None

            def load_to_device_per_layer(
                self, device_pool, host_indices, device_indices, layer_id, io_backend
            ) -> None:
                pass

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ) -> None:
                pass

            def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
                return torch.empty(0)

            def get_dummy_flat_data_page(self) -> torch.Tensor:
                return torch.empty(0)

            def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
                pass

        host_cache = TraceOnlyHostKVCache.__new__(TraceOnlyHostKVCache)
        host_cache.page_size = 4
        groups = [
            HostKVCacheExtentGroup(
                extent_id=0,
                local_indices=torch.tensor(
                    [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19]
                ),
            ),
            HostKVCacheExtentGroup(
                extent_id=1,
                local_indices=torch.tensor([8, 9, 10, 11, 0, 1, 2, 3]),
            ),
        ]

        with unittest.mock.patch.dict(
            "os.environ",
            {"SGLANG_TEST_HICACHE_TRACE_EXTENT_TRANSFERS": "1"},
        ):
            host_cache._record_extent_transfer_groups(
                "D2H", "direct", "page_first_direct", groups
            )

        record = host_cache._test_extent_transfer_records[-1]
        self.assertEqual(record["page_runs"], 4)
        self.assertEqual(record["max_run_pages"], 2)
        self.assertAlmostEqual(record["avg_run_pages"], 1.25)
        self.assertEqual(record["groups"][0]["page_runs"], 2)
        self.assertEqual(record["groups"][0]["max_run_pages"], 2)
        self.assertEqual(record["groups"][1]["page_runs"], 2)
        self.assertEqual(record["groups"][1]["max_run_pages"], 1)

    def test_extent_transfer_trace_records_bytes_and_summary(self):
        class TraceOnlyHostKVCache(HostKVCache):
            def get_size_per_token(self):
                return 0

            def init_kv_buffer(self):
                return None

            def load_to_device_per_layer(
                self, device_pool, host_indices, device_indices, layer_id, io_backend
            ) -> None:
                pass

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ) -> None:
                pass

            def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
                return torch.empty(0)

            def get_dummy_flat_data_page(self) -> torch.Tensor:
                return torch.empty(0)

            def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
                pass

        host_cache = TraceOnlyHostKVCache.__new__(TraceOnlyHostKVCache)
        host_cache.page_size = 4
        host_cache.size_per_token = 128
        host_cache.layer_num = 2
        groups = [
            HostKVCacheExtentGroup(
                extent_id=0,
                local_indices=torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]),
            ),
        ]

        with unittest.mock.patch.dict(
            "os.environ",
            {"SGLANG_TEST_HICACHE_TRACE_EXTENT_TRANSFERS": "1"},
        ):
            host_cache._record_extent_transfer_groups(
                "D2H", "direct", "page_first_direct", groups
            )
            host_cache._record_extent_transfer_groups(
                "H2D", "direct", "page_first_direct", groups, layer_id=0
            )

        d2h_record, h2d_record = host_cache._test_extent_transfer_records
        self.assertEqual(d2h_record["bytes"], 8 * 128)
        self.assertEqual(d2h_record["groups"][0]["bytes"], 8 * 128)
        self.assertEqual(h2d_record["bytes"], 8 * 64)
        self.assertEqual(h2d_record["groups"][0]["bytes"], 8 * 64)

        summary = host_cache._summarize_extent_transfer_records()
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["total_bytes"], 8 * 128 + 8 * 64)
        self.assertEqual(summary["directions"]["D2H"]["records"], 1)
        self.assertEqual(summary["directions"]["D2H"]["bytes"], 8 * 128)
        self.assertEqual(summary["directions"]["H2D"]["records"], 1)
        self.assertEqual(summary["directions"]["H2D"]["bytes"], 8 * 64)
        self.assertEqual(summary["page_run_records"]["records"], 2)
        self.assertEqual(summary["page_run_records"]["max_run_pages"], 1)
        self.assertAlmostEqual(summary["page_run_records"]["avg_run_pages"], 1.0)

    def test_single_extent_transfer_trace_uses_host_indices_as_extent_zero(self):
        class TraceOnlyHostKVCache(HostKVCache):
            def get_size_per_token(self):
                return 0

            def init_kv_buffer(self):
                return None

            def load_to_device_per_layer(
                self, device_pool, host_indices, device_indices, layer_id, io_backend
            ) -> None:
                pass

            def backup_from_device_all_layer(
                self, device_pool, host_indices, device_indices, io_backend
            ) -> None:
                pass

            def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
                return torch.empty(0)

            def get_dummy_flat_data_page(self) -> torch.Tensor:
                return torch.empty(0)

            def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
                pass

        host_cache = TraceOnlyHostKVCache.__new__(TraceOnlyHostKVCache)
        host_cache.page_size = 4
        host_cache.size_per_token = 128
        host_cache.layer_num = 2
        host_indices = torch.tensor([8, 9, 10, 11, 0, 1, 2, 3])
        device_indices = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27])

        with unittest.mock.patch.dict(
            "os.environ",
            {"SGLANG_TEST_HICACHE_TRACE_EXTENT_TRANSFERS": "1"},
        ):
            host_cache._record_single_extent_transfer(
                "H2D",
                "direct",
                "layer_first",
                host_indices,
                device_indices,
                layer_id=0,
            )

        record = host_cache._test_extent_transfer_records[-1]
        self.assertEqual(record["group_count"], 1)
        self.assertEqual(record["groups"][0]["extent_id"], 0)
        self.assertEqual(record["groups"][0]["page_runs"], 2)
        self.assertEqual(record["bytes"], 8 * 64)

    def test_alloc_does_not_assemble_one_page_from_partial_slots_across_extents(self):
        table = HostKVCacheExtentTable(page_size=4)
        table.add_extent(num_slots=8)
        table.add_extent(num_slots=8)

        allocated = table.alloc(16)
        table.free(torch.tensor([2, 3, 8, 9]))

        self.assertIsNone(table.alloc(4))

        table.free(torch.tensor([0, 1]))
        page = table.alloc(4)

        self.assertTrue(torch.equal(page, torch.tensor([0, 1, 2, 3])))
        self.assertEqual(table.page_extent_id(page), 0)

    def test_groups_complete_pages_by_extent_and_rejects_non_page_indices(self):
        table = HostKVCacheExtentTable(page_size=4)
        table.add_extent(num_slots=8)
        table.add_extent(num_slots=8)

        host_indices = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
        device_indices = torch.tensor([40, 41, 42, 43, 44, 45, 46, 47])

        groups = table.group_pages_by_extent(host_indices, paired_indices=device_indices)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].extent_id, 0)
        self.assertTrue(torch.equal(groups[0].local_indices, torch.tensor([4, 5, 6, 7])))
        self.assertTrue(torch.equal(groups[0].paired_indices, torch.tensor([40, 41, 42, 43])))
        self.assertEqual(groups[1].extent_id, 1)
        self.assertTrue(torch.equal(groups[1].local_indices, torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(groups[1].paired_indices, torch.tensor([44, 45, 46, 47])))

        with self.assertRaises(AssertionError):
            table.group_pages_by_extent(torch.tensor([2, 3, 4, 5]))

        with self.assertRaises(AssertionError):
            table.group_pages_by_extent(torch.tensor([0, 1, 2]))

    def test_page_descriptors_use_extent_local_offsets_for_page_id_layouts(self):
        table = HostKVCacheExtentTable(page_size=4)
        table.add_extent(num_slots=8)
        table.add_extent(num_slots=8)

        descriptors = table.page_descriptors(torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]))

        self.assertEqual(len(descriptors), 2)
        self.assertEqual(descriptors[0].extent_id, 0)
        self.assertEqual(descriptors[0].global_start, 4)
        self.assertEqual(descriptors[0].local_start, 4)
        self.assertEqual(descriptors[0].local_page_id, 1)

        self.assertEqual(descriptors[1].extent_id, 1)
        self.assertEqual(descriptors[1].global_start, 8)
        self.assertEqual(descriptors[1].local_start, 0)
        self.assertEqual(descriptors[1].local_page_id, 0)

        with self.assertRaises(AssertionError):
            table.page_descriptors(torch.tensor([6, 7, 8, 9]))

    def test_get_data_page_slices_extent_local_tensor_for_page_id_layouts(self):
        table = HostKVCacheExtentTable(page_size=4)
        extent0_buffer = torch.arange(2 * 2 * 1 * 4 * 1 * 1).reshape(2, 2, 1, 4, 1, 1)
        extent1_buffer = 1000 + torch.arange(2 * 2 * 1 * 4 * 1 * 1).reshape(
            2, 2, 1, 4, 1, 1
        )
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)

        page_from_extent0 = table.get_data_page(4, layout="page_first_direct")
        page_from_extent1 = table.get_data_page(8, layout="page_first_direct")

        self.assertTrue(torch.equal(page_from_extent0, extent0_buffer[:, 1:2].flatten()))
        self.assertTrue(torch.equal(page_from_extent1, extent1_buffer[:, 0:1].flatten()))

    def test_set_from_flat_data_page_writes_extent_local_tensor_for_layer_first(self):
        table = HostKVCacheExtentTable(page_size=4)
        extent0_buffer = torch.zeros((2, 1, 8, 1, 1), dtype=torch.int64)
        extent1_buffer = torch.zeros((2, 1, 8, 1, 1), dtype=torch.int64)
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)

        data_page = torch.arange(2 * 1 * 4 * 1 * 1, dtype=torch.int64)
        table.set_from_flat_data_page(8, data_page, layout="layer_first")

        self.assertTrue(torch.equal(extent1_buffer[:, :, 0:4, :, :].flatten(), data_page))
        self.assertTrue(torch.equal(extent0_buffer, torch.zeros_like(extent0_buffer)))

    def test_set_and_get_data_page_use_extent_local_token_offsets_for_page_first(self):
        table = HostKVCacheExtentTable(page_size=4)
        extent0_buffer = torch.zeros((2, 8, 1, 1, 1), dtype=torch.int64)
        extent1_buffer = torch.zeros((2, 8, 1, 1, 1), dtype=torch.int64)
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)

        data_page = 2000 + torch.arange(2 * 4 * 1 * 1 * 1, dtype=torch.int64)
        table.set_from_flat_data_page(8, data_page, layout="page_first")
        restored = table.get_data_page(8, layout="page_first")

        self.assertTrue(torch.equal(restored, data_page))
        self.assertTrue(torch.equal(extent1_buffer[:, 0:4, :, :, :].flatten(), data_page))
        self.assertTrue(torch.equal(extent0_buffer, torch.zeros_like(extent0_buffer)))

    def test_page_buffer_meta_uses_extent_local_offsets_for_page_first_layouts(self):
        table = HostKVCacheExtentTable(page_size=4)
        extent0_buffer = torch.empty((2, 2, 2, 4, 1, 1), dtype=torch.float32)
        extent1_buffer = torch.empty((2, 2, 2, 4, 1, 1), dtype=torch.float32)
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)

        ptrs, sizes = table.get_page_buffer_meta(
            torch.tensor([8, 9, 10, 11]),
            layout="page_first_direct",
            layer_num=2,
            head_num=1,
            head_dim=1,
        )

        item_size = extent1_buffer.element_size()
        v_offset = 2 * 8 * 1 * 1 * item_size
        self.assertEqual(ptrs, [extent1_buffer.data_ptr(), extent1_buffer.data_ptr() + v_offset])
        self.assertEqual(sizes, [2 * item_size * 4 * 1 * 1, 2 * item_size * 4 * 1 * 1])

    def test_page_buffer_meta_uses_extent_local_offsets_for_layer_first_layout(self):
        table = HostKVCacheExtentTable(page_size=4)
        extent0_buffer = torch.empty((2, 2, 8, 1, 1), dtype=torch.float32)
        extent1_buffer = torch.empty((2, 2, 8, 1, 1), dtype=torch.float32)
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)

        ptrs, sizes = table.get_page_buffer_meta(
            torch.tensor([8, 9, 10, 11]),
            layout="layer_first",
            layer_num=2,
            head_num=1,
            head_dim=1,
        )

        item_size = extent1_buffer.element_size()
        layer_stride = 8 * 1 * 1 * item_size
        v_offset = 2 * 8 * 1 * 1 * item_size
        self.assertEqual(
            ptrs,
            [
                extent1_buffer.data_ptr(),
                extent1_buffer.data_ptr() + v_offset,
                extent1_buffer.data_ptr() + layer_stride,
                extent1_buffer.data_ptr() + layer_stride + v_offset,
            ],
        )
        self.assertEqual(sizes, [item_size * 4 * 1 * 1] * 4)

    def test_dispatch_transfer_groups_passes_extent_buffer_and_local_indices(self):
        table = HostKVCacheExtentTable(page_size=4)
        extent0_buffer = torch.full((2, 8, 1, 1, 1), 10, dtype=torch.int64)
        extent1_buffer = torch.full((2, 8, 1, 1, 1), 20, dtype=torch.int64)
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)

        records = []

        def fake_transfer(kv_buffer, local_host_indices, device_indices, extent_id):
            records.append(
                (
                    extent_id,
                    kv_buffer.data_ptr(),
                    local_host_indices.clone(),
                    device_indices.clone(),
                )
            )

        table.dispatch_transfer_groups(
            host_indices=torch.tensor([2, 8, 3, 9, 10, 4]),
            device_indices=torch.tensor([20, 21, 22, 23, 24, 25]),
            transfer_fn=fake_transfer,
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0], 0)
        self.assertEqual(records[0][1], extent0_buffer.data_ptr())
        self.assertTrue(torch.equal(records[0][2], torch.tensor([2, 3, 4])))
        self.assertTrue(torch.equal(records[0][3], torch.tensor([20, 22, 25])))
        self.assertEqual(records[1][0], 1)
        self.assertEqual(records[1][1], extent1_buffer.data_ptr())
        self.assertTrue(torch.equal(records[1][2], torch.tensor([0, 1, 2])))
        self.assertTrue(torch.equal(records[1][3], torch.tensor([21, 23, 24])))

    def test_cpu_reference_transfer_roundtrip_preserves_cross_extent_index_mapping(self):
        table = HostKVCacheExtentTable(page_size=4)
        extent0_buffer = torch.zeros((2, 8, 1, 1, 1), dtype=torch.int64)
        extent1_buffer = torch.zeros((2, 8, 1, 1, 1), dtype=torch.int64)
        table.add_extent(num_slots=8, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=8, kv_buffer=extent1_buffer)

        device_buffer = torch.arange(2 * 32 * 1 * 1 * 1, dtype=torch.int64).reshape(
            2, 32, 1, 1, 1
        )
        host_indices = torch.tensor([2, 8, 3, 9, 10, 4])
        device_indices = torch.tensor([20, 21, 22, 23, 24, 25])

        table.copy_from_device(
            device_buffer=device_buffer,
            host_indices=host_indices,
            device_indices=device_indices,
            layout="page_first",
        )

        self.assertTrue(
            torch.equal(
                extent0_buffer[:, torch.tensor([2, 3, 4]), :, :, :],
                device_buffer[:, torch.tensor([20, 22, 25]), :, :, :],
            )
        )
        self.assertTrue(
            torch.equal(
                extent1_buffer[:, torch.tensor([0, 1, 2]), :, :, :],
                device_buffer[:, torch.tensor([21, 23, 24]), :, :, :],
            )
        )

        restored_device_buffer = torch.full_like(device_buffer, -1)
        table.copy_to_device(
            device_buffer=restored_device_buffer,
            host_indices=host_indices,
            device_indices=device_indices,
            layout="page_first",
        )

        self.assertTrue(
            torch.equal(
                restored_device_buffer[:, device_indices, :, :, :],
                device_buffer[:, device_indices, :, :, :],
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for kvcacheio")
    def test_real_kvcacheio_roundtrip_preserves_cross_extent_index_mapping(self):
        from sgl_kernel.kvcacheio import (
            transfer_kv_all_layer_lf_pf,
            transfer_kv_per_layer_pf_lf,
        )

        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dtype = torch.float32
        page_size = 4
        extent_size = 8
        layer_num = 2
        device_size = 32
        head_num = 1
        head_dim = 2
        item_size = head_num * head_dim * torch.tensor([], dtype=dtype).element_size()
        layout_dim = item_size * layer_num

        table = HostKVCacheExtentTable(page_size=page_size)
        extent0_buffer = torch.empty(
            (2, extent_size, layer_num, head_num, head_dim),
            dtype=dtype,
            pin_memory=True,
        )
        extent1_buffer = torch.empty_like(extent0_buffer, pin_memory=True)
        extent0_buffer.zero_()
        extent1_buffer.zero_()
        table.add_extent(num_slots=extent_size, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=extent_size, kv_buffer=extent1_buffer)

        dev_k_layers = []
        dev_v_layers = []
        for layer_id in range(layer_num):
            k = torch.zeros((device_size, head_num, head_dim), dtype=dtype, device=device)
            v = torch.zeros_like(k)
            for idx in range(device_size):
                k[idx, 0, :] = 1000 * (layer_id + 1) + idx * 10 + torch.arange(
                    head_dim, device=device
                )
                v[idx, 0, :] = 2000 * (layer_id + 1) + idx * 10 + torch.arange(
                    head_dim, device=device
                )
            dev_k_layers.append(k)
            dev_v_layers.append(v)

        dev_k_ptrs = torch.tensor(
            [x.data_ptr() for x in dev_k_layers], dtype=torch.uint64, device=device
        )
        dev_v_ptrs = torch.tensor(
            [x.data_ptr() for x in dev_v_layers], dtype=torch.uint64, device=device
        )
        host_indices = torch.tensor([2, 8, 3, 9, 10, 4], dtype=torch.int64, device=device)
        device_indices = torch.tensor(
            [20, 21, 22, 23, 24, 25], dtype=torch.int64, device=device
        )

        def backup_to_extent(kv_buffer, local_host_indices, paired_device_indices, _):
            transfer_kv_all_layer_lf_pf(
                src_k_layers=dev_k_ptrs,
                dst_k=kv_buffer[0],
                src_v_layers=dev_v_ptrs,
                dst_v=kv_buffer[1],
                src_indices=paired_device_indices,
                dst_indices=local_host_indices,
                item_size=item_size,
                dst_layout_dim=layout_dim,
                num_layers=layer_num,
            )

        table.dispatch_transfer_groups(host_indices, device_indices, backup_to_extent)
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(
                extent0_buffer[0, torch.tensor([2, 3, 4])].cuda(),
                torch.stack(
                    [
                        dev_k_layers[0][torch.tensor([20, 22, 25], device=device)],
                        dev_k_layers[1][torch.tensor([20, 22, 25], device=device)],
                    ],
                    dim=1,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                extent1_buffer[0, torch.tensor([0, 1, 2])].cuda(),
                torch.stack(
                    [
                        dev_k_layers[0][torch.tensor([21, 23, 24], device=device)],
                        dev_k_layers[1][torch.tensor([21, 23, 24], device=device)],
                    ],
                    dim=1,
                ),
            )
        )

        restored_k_layers = [torch.full_like(x, -1) for x in dev_k_layers]
        restored_v_layers = [torch.full_like(x, -1) for x in dev_v_layers]

        def restore_from_extent(kv_buffer, local_host_indices, paired_device_indices, _):
            for layer_id in range(layer_num):
                transfer_kv_per_layer_pf_lf(
                    src_k=kv_buffer[0],
                    dst_k=restored_k_layers[layer_id],
                    src_v=kv_buffer[1],
                    dst_v=restored_v_layers[layer_id],
                    src_indices=local_host_indices,
                    dst_indices=paired_device_indices,
                    layer_id=layer_id,
                    item_size=item_size,
                    src_layout_dim=layout_dim,
                )

        table.dispatch_transfer_groups(host_indices, device_indices, restore_from_extent)
        torch.cuda.synchronize()

        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(
                    restored_k_layers[layer_id][device_indices],
                    dev_k_layers[layer_id][device_indices],
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for kvcacheio")
    def test_real_kvcacheio_page_head_roundtrip_preserves_cross_extent_mapping(self):
        from sgl_kernel.kvcacheio import (
            transfer_kv_all_layer_lf_ph,
            transfer_kv_per_layer_ph_lf,
        )

        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dtype = torch.float32
        page_size = 4
        extent_size = 8
        page_num = extent_size // page_size
        layer_num = 2
        device_size = 32
        head_num = 2
        head_dim = 2
        item_size = head_num * head_dim * torch.tensor([], dtype=dtype).element_size()
        layout_dim = item_size * layer_num

        table = HostKVCacheExtentTable(page_size=page_size)
        extent0_buffer = torch.empty(
            (2, page_num, head_num, page_size, layer_num, head_dim),
            dtype=dtype,
            pin_memory=True,
        )
        extent1_buffer = torch.empty_like(extent0_buffer, pin_memory=True)
        extent0_buffer.zero_()
        extent1_buffer.zero_()
        table.add_extent(num_slots=extent_size, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=extent_size, kv_buffer=extent1_buffer)

        dev_k_layers = []
        dev_v_layers = []
        for layer_id in range(layer_num):
            k = torch.zeros((device_size, head_num, head_dim), dtype=dtype, device=device)
            v = torch.zeros_like(k)
            for idx in range(device_size):
                for head_id in range(head_num):
                    k[idx, head_id, :] = (
                        1000 * (layer_id + 1)
                        + idx * 10
                        + head_id * head_dim
                        + torch.arange(head_dim, device=device)
                    )
                    v[idx, head_id, :] = (
                        2000 * (layer_id + 1)
                        + idx * 10
                        + head_id * head_dim
                        + torch.arange(head_dim, device=device)
                    )
            dev_k_layers.append(k)
            dev_v_layers.append(v)

        dev_k_ptrs = torch.tensor(
            [x.data_ptr() for x in dev_k_layers], dtype=torch.uint64, device=device
        )
        dev_v_ptrs = torch.tensor(
            [x.data_ptr() for x in dev_v_layers], dtype=torch.uint64, device=device
        )
        host_indices = torch.tensor([2, 8, 3, 9, 10, 4], dtype=torch.int64, device=device)
        device_indices = torch.tensor(
            [20, 21, 22, 23, 24, 25], dtype=torch.int64, device=device
        )

        def backup_to_extent(kv_buffer, local_host_indices, paired_device_indices, _):
            transfer_kv_all_layer_lf_ph(
                src_k_layers=dev_k_ptrs,
                dst_k=kv_buffer[0],
                src_v_layers=dev_v_ptrs,
                dst_v=kv_buffer[1],
                src_indices=paired_device_indices,
                dst_indices=local_host_indices,
                item_size=item_size,
                dst_layout_dim=layout_dim,
                num_layers=layer_num,
                page_size=page_size,
                head_num=head_num,
            )

        table.dispatch_transfer_groups(host_indices, device_indices, backup_to_extent)
        torch.cuda.synchronize()

        restored_k_layers = [torch.full_like(x, -1) for x in dev_k_layers]
        restored_v_layers = [torch.full_like(x, -1) for x in dev_v_layers]

        def restore_from_extent(kv_buffer, local_host_indices, paired_device_indices, _):
            for layer_id in range(layer_num):
                transfer_kv_per_layer_ph_lf(
                    src_k=kv_buffer[0],
                    dst_k=restored_k_layers[layer_id],
                    src_v=kv_buffer[1],
                    dst_v=restored_v_layers[layer_id],
                    src_indices=local_host_indices,
                    dst_indices=paired_device_indices,
                    layer_id=layer_id,
                    item_size=item_size,
                    src_layout_dim=layout_dim,
                    page_size=page_size,
                    head_num=head_num,
                )

        table.dispatch_transfer_groups(host_indices, device_indices, restore_from_extent)
        torch.cuda.synchronize()

        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(
                    restored_k_layers[layer_id][device_indices],
                    dev_k_layers[layer_id][device_indices],
                )
            )
            self.assertTrue(
                torch.equal(
                    restored_v_layers[layer_id][device_indices],
                    dev_v_layers[layer_id][device_indices],
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for kvcacheio")
    def test_mha_production_transfer_methods_dispatch_page_first_extents(self):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dtype = torch.float32
        page_size = 4
        extent_size = 8
        layer_num = 2
        device_size = 32
        head_num = 1
        head_dim = 2

        host_cache = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host_cache.layout = "page_first"
        host_cache.page_size = page_size
        host_cache.layer_num = layer_num
        host_cache.head_num = head_num
        host_cache.head_dim = head_dim
        host_cache.dtype = dtype
        host_cache.token_stride_size = (
            head_num * head_dim * torch.tensor([], dtype=dtype).element_size()
        )
        host_cache.layout_dim = host_cache.token_stride_size * layer_num

        table = HostKVCacheExtentTable(page_size=page_size)
        extent0_buffer = torch.empty(
            (2, extent_size, layer_num, head_num, head_dim),
            dtype=dtype,
            pin_memory=True,
        )
        extent1_buffer = torch.empty_like(extent0_buffer, pin_memory=True)
        extent0_buffer.zero_()
        extent1_buffer.zero_()
        table.add_extent(num_slots=extent_size, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=extent_size, kv_buffer=extent1_buffer)
        host_cache.extent_table = table

        dev_k_layers = []
        dev_v_layers = []
        for layer_id in range(layer_num):
            k = torch.zeros((device_size, head_num, head_dim), dtype=dtype, device=device)
            v = torch.zeros_like(k)
            for idx in range(device_size):
                k[idx, 0, :] = 1000 * (layer_id + 1) + idx * 10 + torch.arange(
                    head_dim, device=device
                )
                v[idx, 0, :] = 2000 * (layer_id + 1) + idx * 10 + torch.arange(
                    head_dim, device=device
                )
            dev_k_layers.append(k)
            dev_v_layers.append(v)

        device_pool = SimpleNamespace(
            k_buffer=dev_k_layers,
            v_buffer=dev_v_layers,
            k_data_ptrs=torch.tensor(
                [x.data_ptr() for x in dev_k_layers], dtype=torch.uint64, device=device
            ),
            v_data_ptrs=torch.tensor(
                [x.data_ptr() for x in dev_v_layers], dtype=torch.uint64, device=device
            ),
        )
        host_indices = torch.tensor([2, 8, 3, 9, 10, 4], dtype=torch.int64, device=device)
        device_indices = torch.tensor(
            [20, 21, 22, 23, 24, 25], dtype=torch.int64, device=device
        )

        host_cache.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend="kernel"
        )
        torch.cuda.synchronize()

        restored_k_layers = [torch.full_like(x, -1) for x in dev_k_layers]
        restored_v_layers = [torch.full_like(x, -1) for x in dev_v_layers]
        restored_pool = SimpleNamespace(
            k_buffer=restored_k_layers,
            v_buffer=restored_v_layers,
        )
        for layer_id in range(layer_num):
            host_cache.load_to_device_per_layer(
                restored_pool,
                host_indices,
                device_indices,
                layer_id=layer_id,
                io_backend="kernel",
            )
        torch.cuda.synchronize()

        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(
                    restored_k_layers[layer_id][device_indices],
                    dev_k_layers[layer_id][device_indices],
                )
            )
            self.assertTrue(
                torch.equal(
                    restored_v_layers[layer_id][device_indices],
                    dev_v_layers[layer_id][device_indices],
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for kvcacheio")
    def test_mha_production_direct_transfer_methods_dispatch_page_first_direct_extents(self):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dtype = torch.float32
        page_size = 4
        extent_size = 8
        page_num = extent_size // page_size
        layer_num = 2
        device_size = 32
        head_num = 1
        head_dim = 2

        host_cache = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host_cache.layout = "page_first_direct"
        host_cache.page_size = page_size
        host_cache.layer_num = layer_num
        host_cache.head_num = head_num
        host_cache.head_dim = head_dim
        host_cache.dtype = dtype

        table = HostKVCacheExtentTable(page_size=page_size)
        extent0_buffer = torch.empty(
            (2, page_num, layer_num, page_size, head_num, head_dim),
            dtype=dtype,
            pin_memory=True,
        )
        extent1_buffer = torch.empty_like(extent0_buffer, pin_memory=True)
        extent0_buffer.zero_()
        extent1_buffer.zero_()
        table.add_extent(num_slots=extent_size, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=extent_size, kv_buffer=extent1_buffer)
        host_cache.extent_table = table

        dev_k_layers = []
        dev_v_layers = []
        for layer_id in range(layer_num):
            k = torch.zeros((device_size, head_num, head_dim), dtype=dtype, device=device)
            v = torch.zeros_like(k)
            for idx in range(device_size):
                k[idx, 0, :] = 1000 * (layer_id + 1) + idx * 10 + torch.arange(
                    head_dim, device=device
                )
                v[idx, 0, :] = 2000 * (layer_id + 1) + idx * 10 + torch.arange(
                    head_dim, device=device
                )
            dev_k_layers.append(k)
            dev_v_layers.append(v)

        device_pool = SimpleNamespace(
            k_buffer=dev_k_layers,
            v_buffer=dev_v_layers,
        )
        host_indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11], dtype=torch.int64)
        device_indices = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27], dtype=torch.int64)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            host_cache.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="direct"
            )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        restored_k_layers = [torch.full_like(x, -1) for x in dev_k_layers]
        restored_v_layers = [torch.full_like(x, -1) for x in dev_v_layers]
        restored_pool = SimpleNamespace(
            k_buffer=restored_k_layers,
            v_buffer=restored_v_layers,
        )
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for layer_id in range(layer_num):
                host_cache.load_to_device_per_layer(
                    restored_pool,
                    host_indices,
                    device_indices,
                    layer_id=layer_id,
                    io_backend="direct",
                )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(
                    restored_k_layers[layer_id][device_indices.to(device)],
                    dev_k_layers[layer_id][device_indices.to(device)],
                )
            )
            self.assertTrue(
                torch.equal(
                    restored_v_layers[layer_id][device_indices.to(device)],
                    dev_v_layers[layer_id][device_indices.to(device)],
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for kvcacheio")
    def test_mha_production_transfer_methods_dispatch_page_head_extents(self):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dtype = torch.float32
        page_size = 4
        extent_size = 8
        page_num = extent_size // page_size
        layer_num = 2
        device_size = 32
        head_num = 2
        head_dim = 2

        host_cache = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host_cache.layout = "page_head"
        host_cache.page_size = page_size
        host_cache.layer_num = layer_num
        host_cache.head_num = head_num
        host_cache.head_dim = head_dim
        host_cache.dtype = dtype
        host_cache.token_stride_size = (
            head_num * head_dim * torch.tensor([], dtype=dtype).element_size()
        )
        host_cache.layout_dim = host_cache.token_stride_size * layer_num

        table = HostKVCacheExtentTable(page_size=page_size)
        extent0_buffer = torch.empty(
            (2, page_num, head_num, page_size, layer_num, head_dim),
            dtype=dtype,
            pin_memory=True,
        )
        extent1_buffer = torch.empty_like(extent0_buffer, pin_memory=True)
        extent0_buffer.zero_()
        extent1_buffer.zero_()
        table.add_extent(num_slots=extent_size, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=extent_size, kv_buffer=extent1_buffer)
        host_cache.extent_table = table

        dev_k_layers = []
        dev_v_layers = []
        for layer_id in range(layer_num):
            k = torch.zeros((device_size, head_num, head_dim), dtype=dtype, device=device)
            v = torch.zeros_like(k)
            for idx in range(device_size):
                for head_id in range(head_num):
                    k[idx, head_id, :] = (
                        1000 * (layer_id + 1)
                        + idx * 10
                        + head_id * head_dim
                        + torch.arange(head_dim, device=device)
                    )
                    v[idx, head_id, :] = (
                        2000 * (layer_id + 1)
                        + idx * 10
                        + head_id * head_dim
                        + torch.arange(head_dim, device=device)
                    )
            dev_k_layers.append(k)
            dev_v_layers.append(v)

        device_pool = SimpleNamespace(
            k_buffer=dev_k_layers,
            v_buffer=dev_v_layers,
            k_data_ptrs=torch.tensor(
                [x.data_ptr() for x in dev_k_layers], dtype=torch.uint64, device=device
            ),
            v_data_ptrs=torch.tensor(
                [x.data_ptr() for x in dev_v_layers], dtype=torch.uint64, device=device
            ),
        )
        host_indices = torch.tensor([2, 8, 3, 9, 10, 4], dtype=torch.int64, device=device)
        device_indices = torch.tensor(
            [20, 21, 22, 23, 24, 25], dtype=torch.int64, device=device
        )

        host_cache.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend="kernel"
        )
        torch.cuda.synchronize()

        restored_k_layers = [torch.full_like(x, -1) for x in dev_k_layers]
        restored_v_layers = [torch.full_like(x, -1) for x in dev_v_layers]
        restored_pool = SimpleNamespace(
            k_buffer=restored_k_layers,
            v_buffer=restored_v_layers,
        )
        for layer_id in range(layer_num):
            host_cache.load_to_device_per_layer(
                restored_pool,
                host_indices,
                device_indices,
                layer_id=layer_id,
                io_backend="kernel",
            )
        torch.cuda.synchronize()

        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(
                    restored_k_layers[layer_id][device_indices],
                    dev_k_layers[layer_id][device_indices],
                )
            )
            self.assertTrue(
                torch.equal(
                    restored_v_layers[layer_id][device_indices],
                    dev_v_layers[layer_id][device_indices],
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for kvcacheio")
    def test_mla_production_transfer_methods_dispatch_page_first_extents(self):
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        dtype = torch.float32
        page_size = 4
        extent_size = 8
        layer_num = 2
        device_size = 32
        kv_cache_dim = 2

        host_cache = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host_cache.layout = "page_first"
        host_cache.page_size = page_size
        host_cache.layer_num = layer_num
        host_cache.kv_cache_dim = kv_cache_dim
        host_cache.dtype = dtype
        host_cache.token_stride_size = (
            kv_cache_dim * torch.tensor([], dtype=dtype).element_size()
        )
        host_cache.layout_dim = host_cache.token_stride_size * layer_num

        table = HostKVCacheExtentTable(page_size=page_size)
        extent0_buffer = torch.empty(
            (extent_size, layer_num, 1, kv_cache_dim),
            dtype=dtype,
            pin_memory=True,
        )
        extent1_buffer = torch.empty_like(extent0_buffer, pin_memory=True)
        extent0_buffer.zero_()
        extent1_buffer.zero_()
        table.add_extent(num_slots=extent_size, kv_buffer=extent0_buffer)
        table.add_extent(num_slots=extent_size, kv_buffer=extent1_buffer)
        host_cache.extent_table = table

        dev_layers = []
        for layer_id in range(layer_num):
            layer = torch.zeros((device_size, 1, kv_cache_dim), dtype=dtype, device=device)
            for idx in range(device_size):
                layer[idx, 0, :] = 1000 * (layer_id + 1) + idx * 10 + torch.arange(
                    kv_cache_dim, device=device
                )
            dev_layers.append(layer)

        device_pool = SimpleNamespace(
            kv_buffer=dev_layers,
            data_ptrs=torch.tensor(
                [x.data_ptr() for x in dev_layers], dtype=torch.uint64, device=device
            ),
        )
        host_indices = torch.tensor([2, 8, 3, 9, 10, 4], dtype=torch.int64, device=device)
        device_indices = torch.tensor(
            [20, 21, 22, 23, 24, 25], dtype=torch.int64, device=device
        )

        host_cache.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend="kernel"
        )
        torch.cuda.synchronize()

        restored_layers = [torch.full_like(x, -1) for x in dev_layers]
        restored_pool = SimpleNamespace(kv_buffer=restored_layers)
        for layer_id in range(layer_num):
            host_cache.load_to_device_per_layer(
                restored_pool,
                host_indices,
                device_indices,
                layer_id=layer_id,
                io_backend="kernel",
            )
        torch.cuda.synchronize()

        for layer_id in range(layer_num):
            self.assertTrue(
                torch.equal(
                    restored_layers[layer_id][device_indices],
                    dev_layers[layer_id][device_indices],
                )
            )


if __name__ == "__main__":
    unittest.main()
