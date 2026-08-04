import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_tier1_v2 as bt


def _cfg(**kw):
    defaults = dict(name="test", k_p=1, k_d=1, tp_pa=1, tp_pf=1, tp_da=1,
                    tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=930, tier=False)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_plan_allocation_rejects_unplaced_prefill_pairs():
    cfg = _cfg(k_p=8, k_d=1)
    try:
        bt.plan_allocation(cfg)
    except RuntimeError as exc:
        assert "Cannot place all P pairs" in str(exc)
    else:
        raise AssertionError("expected allocation failure")


def test_router_command_contains_every_decode():
    cfg = _cfg(k_p=1, k_d=2)
    commands = []
    with patch.object(bt, "plan_allocation") as plan, \
         patch.object(bt.RMB, "_launch"), patch.object(bt.RMB, "wait_health", return_value=True), \
         patch.object(bt.RMB, "dexec_local", side_effect=lambda c: commands.append(c)), \
         patch.object(bt.RMB, "dexec_remote"), patch.object(bt.RMB, "write_ib_json"), \
         patch.object(bt.time, "sleep"):
        plan.return_value = {"p_pairs": [(bt.RMB.NODE1_IP, [1], [0])],
            "decode": {"host": bt.RMB.NODE1_IP, "attn": [3], "ffn": [2]},
            "decode_instances": [
                {"host": bt.RMB.NODE1_IP, "attn": [3], "ffn": [2]},
                {"host": bt.RMB.NODE2_IP, "attn": [1], "ffn": [0]}],
            "freq_map": {bt.RMB.NODE1_IP: {}, bt.RMB.NODE2_IP: {}}, "total_gpu": 6}
        urls = bt.deploy(cfg)
    router = next(c for c in commands if "launch_router" in c)
    assert router.count("--decode ") == 2
    assert urls == [f"http://{bt.RMB.NODE1_IP}:{bt.SUB_ROUTER_BASE}"]


def test_partial_completion_is_not_pass():
    reqs = [{"arrival_time_s": 0, "_target_url": "http://r/generate"} for _ in range(2)]
    async def send_one(session, url, req, base_time, results):
        if not results:
            results.append({"success": True, "completion_tokens": 1, "ttft_ms": 1,
                            "ttft_proc_ms": 1, "tpot_ms": 1})
        else:
            await asyncio.sleep(1)
    with patch.object(bt.RMB, "send_one", side_effect=send_one), \
         patch.object(bt.RMB, "get_energy_local", return_value={}), \
         patch.object(bt.RMB, "get_energy_remote", return_value={}):
        result = asyncio.run(bt._run_workload_rr(reqs, [], [], max_run_s=0.05))
    assert result["status"] == "PARTIAL_TIMEOUT"
    assert result["successful"] == 1
    assert result["timed_out"] == 1


def test_weighted_prefill_assignment():
    cfg = bt.Tier1TestConfig(
        name="mixed", k_p=3, k_d=3,
        tp_pa=2, tp_pf=2, tp_da=1, tp_df=1,
        f_pa=1410, f_pf=1410, f_da=930, f_df=930,
        prefill_specs=(
            bt.PrefillSpec(2, 2, 1410, 1410),
            bt.PrefillSpec(2, 2, 1410, 1410),
            bt.PrefillSpec(1, 1, 930, 930),
        ),
    )
    urls = ["http://r0", "http://r1", "http://r2"]
    assigned = bt.assign_prefill_urls(urls, cfg, 10)
    assert assigned == [
        "http://r0/generate", "http://r0/generate",
        "http://r1/generate", "http://r1/generate",
        "http://r2/generate",
        "http://r0/generate", "http://r0/generate",
        "http://r1/generate", "http://r1/generate",
        "http://r2/generate",
    ]


    reqs = [{"arrival_time_s": 0, "_target_url": "http://r/generate"}]
    async def send_one(*args):
        await asyncio.sleep(1)
    with patch.object(bt.RMB, "send_one", side_effect=send_one), \
         patch.object(bt.RMB, "get_energy_local", return_value={}), \
         patch.object(bt.RMB, "get_energy_remote", return_value={}):
        result = asyncio.run(bt._run_workload_rr(reqs, [], [], max_run_s=0.05))
    assert result["status"] == "TIMEOUT"
    assert result["successful"] == 0
