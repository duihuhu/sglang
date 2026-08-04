import unittest
from dataclasses import replace
import torch
from sglang.srt.layers.afd_mixin import AFDWeightFilter
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.reshard.afd_weight_reshard import (
    AFDWeightReshardTransaction, Ownership, ReshardValidationError, SplitRule,
    TransactionState, build_manifest, build_staging_plan, reshard_tensor_shards,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
register_cpu_ci(est_time=2, suite="stage-a-cpu-only")

def _shard(full, rule, tp):
    if rule.kind == "replicated": return tuple(full.clone() for _ in range(tp))
    if rule.kind == "column_fused":
        out = []
        for rank in range(tp):
            off, pieces = 0, []
            for size in rule.segments:
                pieces.append(full.narrow(0, off + rank * (size // tp), size // tp)); off += size
            out.append(torch.cat(pieces, 0).clone())
        return tuple(out)
    size = full.shape[rule.dim] // tp
    return tuple(full.narrow(rule.dim, rank * size, size).clone() for rank in range(tp))

class TestAFDWeightReshard(CustomTestCase):
    def test_a2_to_a4_column_row_replicated_and_transaction(self):
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(64).reshape(8, 8),
            "model.layers.0.self_attn.o_proj.weight": torch.arange(64).reshape(8, 8),
            "model.norm.weight": torch.arange(8),
        }
        rules = {list(tensors)[0]: ("column", 0), list(tensors)[1]: ("row", 1)}
        source = {name: _shard(t, SplitRule.from_tp_rule(rules.get(name)), 2) for name, t in tensors.items()}
        manifest = build_manifest("A-expand", 7, AFDPerspective.AFD_PERSPECTIVE_ATTN, 2, 4, source, rules)
        self.assertTrue(all(p.ownership in (Ownership.ATTN, Ownership.SHARED) for p in manifest.parameters))
        tx = AFDWeightReshardTransaction(manifest); tx.prepare(source, manifest.manifest_hash)
        self.assertEqual(tx.state, TransactionState.READY)
        staged = tx.commit(); self.assertEqual(tx.state, TransactionState.COMMITTED)
        for name, full in tensors.items():
            expected = _shard(full, SplitRule.from_tp_rule(rules.get(name)), 4)
            for rank in range(4): self.assertTrue(torch.equal(staged[rank][name], expected[rank]))
        plan = build_staging_plan(manifest, 7)
        self.assertEqual(plan.total_bytes, sum(p.target_bytes for p in manifest.parameters))
        self.assertTrue(all(c.device == "cpu" and c.length_bytes <= 7 for c in plan.chunks))

    def test_f2_to_f4_gate_up(self):
        name = "model.layers.0.mlp.gate_up_proj.weight"; full = torch.arange(16 * 4).reshape(16, 4)
        rule = SplitRule("column_fused", 0, (8, 8)); source = {name: _shard(full, rule, 2)}
        manifest = build_manifest("F-expand", 1, AFDPerspective.AFD_PERSPECTIVE_FFN, 2, 4, source, {name: ("column_fused", 0, (8, 8))})
        tx = AFDWeightReshardTransaction(manifest); tx.prepare(source)
        for rank, expected in enumerate(_shard(full, rule, 4)): self.assertTrue(torch.equal(tx.shadow_tensors[rank][name], expected))

    def test_a4_to_a1_fused_qkv_is_segment_major(self):
        rule = SplitRule("column_fused", 0, (8, 4, 4)); full = torch.arange(16 * 3).reshape(16, 3)
        source = _shard(full, rule, 4)
        self.assertFalse(torch.equal(torch.cat(source, 0), full))
        self.assertTrue(torch.equal(reshard_tensor_shards(source, rule, 1)[0], full))
        column = torch.arange(32).reshape(8, 4)
        self.assertTrue(torch.equal(reshard_tensor_shards(_shard(column, SplitRule("column", 0), 4), SplitRule("column", 0), 1)[0], column))

    def test_manifest_hash_mismatch_and_checksum(self):
        name = "model.layers.0.self_attn.q_proj.weight"; source = {name: (torch.arange(8).reshape(2, 4),) * 2}
        manifest = build_manifest("hash", 3, AFDPerspective.AFD_PERSPECTIVE_ATTN, 2, 4, source, {name: ("column", 0)}, checksum_mode="strict")
        with self.assertRaisesRegex(ReshardValidationError, "hash mismatch"):
            AFDWeightReshardTransaction(replace(manifest, epoch=4)).prepare(source)
        tx = AFDWeightReshardTransaction(manifest); changed = {name: (source[name][0] + 1, source[name][1])}
        with self.assertRaisesRegex(ReshardValidationError, "checksum mismatch"): tx.prepare(changed)

    def test_ownership_and_should_load_compatibility(self):
        names = {"x.self_attn.q_proj.weight": "attn", "x.mlp.down_proj.weight": "ffn", "x.norm.weight": "shared", "x.rotary_emb.inv_freq": "shared"}
        for name, owner in names.items(): self.assertEqual(AFDWeightFilter.classify(name), owner)
        self.assertFalse(AFDWeightFilter.should_load("x.mlp.up_proj.weight", AFDPerspective.AFD_PERSPECTIVE_ATTN))
        self.assertFalse(AFDWeightFilter.should_load("x.self_attn.o_proj.weight", AFDPerspective.AFD_PERSPECTIVE_FFN))
        self.assertTrue(AFDWeightFilter.should_load("x.unknown", AFDPerspective.AFD_PERSPECTIVE_ATTN))

    def test_prepare_allows_source_superset_and_checks_dtype(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        source = {name: (torch.arange(8).reshape(2, 4),) * 2}
        manifest = build_manifest(
            "superset", 0, AFDPerspective.AFD_PERSPECTIVE_ATTN,
            2, 4, source, {name: ("column", 0)},
        )
        extra = dict(source)
        extra["unused.parameter"] = (torch.ones(1),) * 2
        tx = AFDWeightReshardTransaction(manifest)
        tx.prepare(extra)
        self.assertEqual(tx.state, TransactionState.READY)

        changed = {name: tuple(shard.to(torch.float32) for shard in source[name])}
        with self.assertRaisesRegex(ReshardValidationError, "dtype changed"):
            AFDWeightReshardTransaction(manifest).prepare(changed)

    def test_manifest_recomputes_bytes_and_staging_divisibility(self):
        name = "model.norm.weight"
        source = {name: (torch.arange(4, dtype=torch.float32),) * 2}
        manifest = build_manifest(
            "bytes", 0, AFDPerspective.AFD_PERSPECTIVE_ATTN,
            2, 3, source, {},
        )
        self.assertEqual(manifest.parameters[0].source_bytes, 32)
        self.assertEqual(manifest.parameters[0].target_bytes, 48)
        tampered_entry = replace(manifest.parameters[0], target_bytes=47)
        tampered = replace(manifest, parameters=(tampered_entry,), manifest_hash="")
        tampered = replace(tampered, manifest_hash=tampered.computed_hash())
        with self.assertRaisesRegex(ReshardValidationError, "target byte invariant"):
            build_staging_plan(tampered)

    def test_strict_ownership_and_unknown_fail_closed(self):
        self.assertEqual(AFDWeightFilter.classify_strict("x.self_attn.q_norm.weight"), "attn")
        self.assertEqual(AFDWeightFilter.classify_strict("x.self_attn.k_norm.weight"), "attn")
        self.assertIsNone(AFDWeightFilter.classify_strict("x.unknown.weight"))
        unknown = {"x.unknown.weight": (torch.ones(2),) * 2}
        with self.assertRaisesRegex(ReshardValidationError, "unknown or ambiguous"):
            build_manifest(
                "unknown", 0, AFDPerspective.AFD_PERSPECTIVE_ATTN,
                2, 4, unknown, {},
            )

    def test_gqa_kv_replication_is_explicitly_rejected(self):
        name = "model.layers.0.self_attn.k_proj.weight"
        source = {name: (torch.ones(2, 2),) * 2}
        with self.assertRaisesRegex(ReshardValidationError, "GQA KV replication"):
            build_manifest(
                "gqa", 0, AFDPerspective.AFD_PERSPECTIVE_ATTN,
                2, 4, source, {name: ("column", 0)},
                kv_replication_metadata={"num_key_value_heads": 2},
            )

    def test_abort_and_explicit_failures(self):
        name = "model.layers.0.mlp.down_proj.weight"; source = {name: (torch.ones(2, 2),) * 2}
        manifest = build_manifest("abort", 1, AFDPerspective.AFD_PERSPECTIVE_FFN, 2, 4, source, {name: ("row", 1)})
        tx = AFDWeightReshardTransaction(manifest); tx.prepare(source); tx.abort("cancelled")
        self.assertEqual(tx.state, TransactionState.ABORTED); self.assertEqual(tx.shadow_tensors, {})
        with self.assertRaises(ReshardValidationError): tx.commit()
        with self.assertRaisesRegex(ReshardValidationError, "unknown TP split rule"):
            build_manifest("bad", 1, AFDPerspective.AFD_PERSPECTIVE_FFN, 2, 4, source, {name: ("mystery", 0)})
        with self.assertRaisesRegex(ReshardValidationError, "quantized"):
            build_manifest("quant", 1, AFDPerspective.AFD_PERSPECTIVE_FFN, 2, 4, {"x.mlp.qweight": source[name]}, {},)
        with self.assertRaisesRegex(ReshardValidationError, "unsupported dense model family"):
            build_manifest("moe", 1, AFDPerspective.AFD_PERSPECTIVE_FFN, 2, 4, source, {name: ("row", 1)}, model_family="mixtral")

if __name__ == "__main__": unittest.main(verbosity=3)
