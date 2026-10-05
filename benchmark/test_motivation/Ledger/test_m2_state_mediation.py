import copy
import unittest

try:
    from run_m2_state_mediation import COUNT, command, connection_audit, fingerprint, preconnect, session_plan, validate_info, workload
except ModuleNotFoundError as error:
    if error.name != "aiohttp":
        raise
    command = None


def info(pool="natural", cap=485):
    state = 485 if pool == "natural" else 728
    return {"internal_states": [{
        "tp_size": 2, "pp_size": 1, "tenant_slot_gate_code": 1, "tenant_new_activation_limit": 0,
        "schedule_policy": "tenant-vtc", "tenant_vtc_request_weight": 0,
        "tenant_vtc_input_weight": 1, "tenant_vtc_output_weight": 2,
        "tenant_vtc_input_quadratic_weight": 0, "tenant_vtc_input_output_weight": 0,
        "tenant_vtc_output_quadratic_weight": 0, "disable_radix_cache": True,
        "chunked_prefill_size": 4096, "enable_mixed_chunk": True,
        "enable_prefill_delayer": False, "min_free_slots_delay": None, "prefill_max_requests": None,
        "cuda_graph_config": {p: {"backend": "disabled"} for p in ("prefill", "decode")},
        "tenant_resource_trace": {"event_trace_enabled": True}, "max_total_tokens": None,
        "memory_usage": {"token_capacity": 1458643 if pool == "natural" else 800000},
        "max_mamba_cache_size": state, "effective_max_running_requests_per_dp": state,
        "pp_max_micro_batch_size": cap, "mem_fraction_static": .85,
        "max_prefill_tokens": 16384, "max_running_requests": None,
        "enable_unified_memory": False, "speculative_algorithm": None,
    }]}


@unittest.skipIf(command is None, "Replay dependencies unavailable locally")
class MediationTests(unittest.TestCase):
    def test_natural_command_does_not_resize_pool_for_runtime_cap(self):
        args = command("mixed-1024", "natural", 30133)
        for flag in ("--max-mamba-cache-size", "--max-total-tokens", "--max-running-requests"):
            self.assertNotIn(flag, args)
        self.assertEqual(args[args.index("--chunked-prefill-size") + 1], "1024")

    def test_wide_pool_keeps_same_fraction_and_auto_kv(self):
        args = command("mixed-4096", "wide", 30133)
        self.assertEqual(args[args.index("--max-mamba-cache-size") + 1], "728")
        self.assertEqual(args[args.index("--mem-fraction-static") + 1], "0.85")
        self.assertNotIn("--max-total-tokens", args)

    def test_full_workload_is_paired_across_conditions(self):
        a = workload("natural-a", 11, "calibrate")
        b = workload("wide-b", 11, "mediate")
        self.assertEqual(len(a), COUNT)
        self.assertEqual(fingerprint(a), fingerprint(b))
        self.assertNotEqual(fingerprint(a), fingerprint(workload("new", 12, "mediate")))
        self.assertTrue(all(len(r.input_ids) == 512 and r.output_len == 128 for r in a))

    def test_full_warmup_is_distinct_and_functional_inspection_is_small(self):
        warm = workload("warm", 11, "warmup")
        self.assertEqual(len(warm), COUNT)
        self.assertNotEqual(fingerprint(warm), fingerprint(workload("main", 11, "calibrate")))
        self.assertEqual(len(workload("check", 1, "inspect")), 3)

    def test_planned_matrix_is_complete(self):
        self.assertEqual(sum(len(p[3]) for p in session_plan("calibrate")), 18)
        self.assertEqual(sum(len(p[3]) for p in session_plan("mediate")), 8)
        self.assertEqual(len(session_plan("inspect")), 2)

    def test_modes_reverse_between_restarts(self):
        for stage in ("calibrate", "mediate"):
            plan = session_plan(stage)
            self.assertEqual([p[1] for p in plan if p[2] == 11],
                             list(reversed([p[1] for p in plan if p[2] == 12])))

    def test_raw_info_preserved_and_runtime_cap_distinct_from_pool(self):
        original = info("wide", 607)
        before = copy.deepcopy(original)
        self.assertEqual(validate_info(original, "mixed-4096", "wide", 607), (728, 800000, 728, 607))
        self.assertEqual(original, before)

    def test_reject_capacity_drift(self):
        with self.assertRaisesRegex(ValueError, "Physical capacity"):
            validate_info(info("wide", 607), "mixed-4096", "wide", 607, {"wide": [728, 800001, 728]})

    def test_reject_wrong_pool_and_cap(self):
        for pool, cap in (("wide", 485), ("natural", 607)):
            with self.assertRaises(ValueError):
                validate_info(info(), "mixed-4096", pool, cap)

    def test_reject_kv_pressure_or_expanded_budget(self):
        for key, value in (("mem_fraction_static", .9), ("enable_unified_memory", True)):
            item = info()
            item["internal_states"][0][key] = value
            with self.assertRaises(ValueError):
                validate_info(item, "mixed-4096", "natural", 485)
        item = info()
        item["internal_states"][0]["memory_usage"]["token_capacity"] = 700000
        with self.assertRaisesRegex(ValueError, "KV headroom"):
            validate_info(item, "mixed-4096", "natural", 485)


@unittest.skipIf(command is None, "Replay dependencies unavailable locally")
class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_preconnect_reuses_connections_without_inference(self):
        import aiohttp
        from aiohttp import web
        app = web.Application()
        seen = []
        async def model_info(request):
            seen.append(request.path)
            return web.json_response({"model_type": "test"})
        app.router.add_get("/model_info", model_info)
        server = web.AppRunner(app)
        await server.setup()
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        try:
            trace, audit = connection_audit()
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), trace_configs=[trace]) as client:
                url = f"http://127.0.0.1:{server.addresses[0][1]}"
                audit["phase"] = "preconnect"
                await preconnect(client, url, 32)
                self.assertEqual(audit["created"]["preconnect"], 32)
                audit["phase"] = "replay"
                await preconnect(client, url, 32)
                self.assertEqual(audit["created"].get("replay", 0), 0)
                self.assertEqual(audit["reused"]["replay"], 32)
            self.assertEqual(seen, ["/model_info"] * 64)
        finally:
            await server.cleanup()


if __name__ == "__main__":
    unittest.main()
