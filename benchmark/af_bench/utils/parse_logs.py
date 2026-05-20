#!/usr/bin/env python3
"""Parse DA/DF logs with AFD_TIMELINE and create detailed M=3 pipeline visualization."""
import json, os, sys, re
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(_HERE, "raw_logs")
OUT = os.path.join(_HERE, "results")
os.makedirs(OUT, exist_ok=True)


def parse_log(filepath):
    """Extract AFD_TIMELINE and AFD_BREAKDOWN entries from a log file."""
    timelines = []
    breakdowns = []
    with open(filepath) as f:
        for line in f:
            if "AFD_TIMELINE" in line:
                # Extract log timestamp
                ts_match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
                log_ts = ts_match.group(1) if ts_match else "unknown"

                # Extract JSON payload
                idx = line.index("timeline=")
                payload = line[idx + len("timeline="):].strip()
                try:
                    tl = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                # Extract metadata
                m = re.search(r"M=(\d+)", line)
                m_val = int(m.group(1)) if m else 1
                perspective = "attn" if "perspective=attn" in line else "ffn"

                timelines.append({
                    "log_ts": log_ts,
                    "perspective": perspective,
                    "M": m_val,
                    "total_steps": len(tl),
                    "steps": tl,
                })

            if "AFD_BREAKDOWN" in line:
                ts_match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
                log_ts = ts_match.group(1) if ts_match else "unknown"
                m = re.search(r"M=(\d+)", line)
                m_val = int(m.group(1)) if m else 1
                perspective = "attn" if "perspective=attn" in line else "ffn"
                breakdowns.append({
                    "log_ts": log_ts,
                    "perspective": perspective,
                    "M": m_val,
                    "raw": line.strip(),
                })
    return timelines, breakdowns


def main():
    da_tl, da_bd = parse_log(os.path.join(RAW, "da.log"))
    df_tl, df_bd = parse_log(os.path.join(RAW, "df.log"))

    print(f"DA: {len(da_tl)} timelines, {len(da_bd)} breakdowns")
    print(f"DF: {len(df_tl)} timelines, {len(df_bd)} breakdowns")

    # Filter M=3 only
    da_m3 = [t for t in da_tl if t["M"] == 3]
    df_m3 = [t for t in df_tl if t["M"] == 3]
    da_bd_m3 = [b for b in da_bd if b["M"] == 3]
    df_bd_m3 = [b for b in df_bd if b["M"] == 3]

    print(f"DA M=3: {len(da_m3)} timelines, {len(da_bd_m3)} breakdowns")
    print(f"DF M=3: {len(df_m3)} timelines, {len(df_bd_m3)} breakdowns")

    # Select a representative M=3 forward pass for detailed analysis
    # Pick the middle one (not warmup, not tail)
    idx = min(len(da_m3) // 2, 5)
    da_sample = da_m3[idx]
    df_sample = df_m3[idx] if idx < len(df_m3) else df_m3[-1]

    print(f"\nUsing DA timeline idx={idx}: log_ts={da_sample['log_ts']}, steps={da_sample['total_steps']}")
    print(f"Using DF timeline idx={idx}: log_ts={df_sample['log_ts']}, steps={df_sample['total_steps']}")

    # Parse breakdown for this sample
    if idx < len(da_bd_m3):
        print(f"\nDA Breakdown: {da_bd_m3[idx]['raw'][:300]}")
    if idx < len(df_bd_m3):
        print(f"DF Breakdown: {df_bd_m3[idx]['raw'][:300]}")

    # Save parsed data as JSON for the visualization script
    output = {
        "da_timeline": da_sample,
        "df_timeline": df_sample,
        "da_breakdown": da_bd_m3[idx].get("raw", "") if idx < len(da_bd_m3) else "",
        "df_breakdown": df_bd_m3[idx].get("raw", "") if idx < len(df_bd_m3) else "",
        "summary": {
            "total_da_m3_passes": len(da_m3),
            "total_df_m3_passes": len(df_m3),
        }
    }

    out_path = os.path.join(OUT, "parsed_timeline_m3.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved parsed data to {out_path}")

    # Also compute aggregated stats from all M=3 breakdowns
    print("\n--- Aggregated M=3 Breakdown Stats ---")
    for label, bds in [("DA", da_bd_m3), ("DF", df_bd_m3)]:
        # Parse breakdown lines for key metrics
        totals = []
        a_stages = []
        f_stages = []
        for b in bds:
            m = re.search(r"total=([\d.]+)ms", b["raw"])
            if m: totals.append(float(m.group(1)))
            m = re.search(r"A_stage=([\d.]+)ms", b["raw"])
            if m: a_stages.append(float(m.group(1)))
            m = re.search(r"F_stage=([\d.]+)ms", b["raw"])
            if m: f_stages.append(float(m.group(1)))

        if totals:
            print(f"  {label}: total={sum(totals)/len(totals):.1f}ms, "
                  f"A_stage={sum(a_stages)/len(a_stages):.1f}ms, "
                  f"F_stage={sum(f_stages)/len(f_stages):.1f}ms "
                  f"(n={len(totals)})")

    # Save summary stats
    stats = {
        "da_m3": {},
        "df_m3": {},
    }
    for label, bds, key in [("DA", da_bd_m3, "da_m3"), ("DF", df_bd_m3, "df_m3")]:
        if not bds:
            continue
        # Parse first breakdown for detailed sub-stage breakdown
        b = bds[idx] if idx < len(bds) else bds[0]
        raw = b["raw"]
        stats[key]["sample_raw"] = raw
        for field in ["total", "A_stage", "F_stage", "prep_attn", "attn", "prep_mlp", "mlp", "postprocess", "nA", "nF"]:
            pattern = rf"{field}=([\d.]+)"
            m = re.search(pattern, raw)
            if m:
                stats[key][field] = float(m.group(1))

    stats_path = os.path.join(OUT, "breakdown_stats_m3.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
