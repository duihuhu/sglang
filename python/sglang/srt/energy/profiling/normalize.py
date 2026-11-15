"""Normalize AF profiles: latency stays us; rank-0 energy stays mJ."""

import argparse
from io import StringIO
from pathlib import Path


def normalize_profile(df, phase):
    """Convert raw headers to V1 without TP, latency, or energy scaling."""
    if phase not in ("prefill", "decode"):
        raise ValueError(f"Unknown phase: {phase}")
    prefix = "P" if phase == "prefill" else "D"
    df = df.copy()
    df.columns = df.columns.str.strip()
    aliases = {
        f"{prefix}_A_lat": "A",
        f"{prefix}_F_lat": "F",
        f"{prefix}_A_energy": "A_energy_mj",
        f"{prefix}_F_energy": "F_energy_mj",
    }
    for raw, canonical in aliases.items():
        if raw in df and canonical in df:
            raise ValueError(f"Ambiguous columns: {raw}, {canonical}")
    df = df.rename(columns=aliases)
    columns = ["tp", "input_len"]
    if phase == "decode":
        columns.append("output_len")
    columns += ["gpu_clock", "batch_size", "A", "F", "A_energy_mj", "F_energy_mj"]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")
    return df[columns].reset_index(drop=True)


def load_profile(path, phase):
    """Read canonical/raw TSV with an optional descriptive preamble."""
    import pandas as pd

    lines = Path(path).read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip().startswith("tp\t"):
            return normalize_profile(
                pd.read_csv(StringIO("".join(lines[i:])), sep="\t"), phase
            )
    raise ValueError(f"No tab-separated tp header in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    df = load_profile(args.input, args.phase)
    with open(args.output, "w") as stream:
        stream.write(f"# {args.phase}: latency_us; rank0_energy_mj; no TP scaling\n")
        df.to_csv(stream, sep="\t", index=False)


if __name__ == "__main__":
    main()
