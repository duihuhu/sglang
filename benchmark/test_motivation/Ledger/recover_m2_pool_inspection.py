"""Create a new plan for unstarted pool inspections; never amend the old run."""

import argparse
import json
from pathlib import Path

from run_m2_pool_comparison import source_hashes
from run_m2_batch_frontier import dump, hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    hashes = source_hashes()
    if manifest["source_hashes"] != hashes or (args.source / "completed.json").exists():
        raise ValueError("Expected an incomplete inspect with unchanged sources")
    adopted = {}
    baseline_ids = None
    for index, arm in enumerate(("static-natural", "unified-old-cap")):
        session = args.source / f"session-{index:02d}-{arm}-mixed-4096"
        artifacts = sorted(session.glob("m2pool-*.json"))
        if len(artifacts) != 2:
            raise ValueError("Expected a full warmup and diagnostic pair")
        for path in artifacts:
            data = json.loads(path.read_text())
            if data["integrity_errors"] or len(data["requests"]) != 607:
                raise ValueError(f"Invalid completed diagnostic {path}")
            adopted[str(path)] = hash_file(path)
        path = session / f"correctness-{arm}.json"
        ids = [r["output_ids"] for r in json.loads(path.read_text())["records"]]
        if baseline_ids is None:
            baseline_ids = ids
        elif ids != baseline_ids:
            raise ValueError("Previous correctness mismatch")
        adopted[str(path)] = hash_file(path)
    dump(args.plan, {
        "source_hashes": hashes,
        "sessions": [{"arm": arm, "mode": "mixed-4096", "reps": [21]}
                     for arm in ("static-rebalanced", "unified-raised-cap")],
        "recovery": {"source_manifest": str(args.source / "manifest.json"),
                     "adopted_diagnostics_sha256": adopted,
                     "cross_pool_correctness_reference": baseline_ids,
                     "script_sha256": hash_file(Path(__file__)),
                     "reason": "Third server failed its TP free-memory-balance check before loading; no requests adopted from failed session."},
    })


if __name__ == "__main__":
    main()
