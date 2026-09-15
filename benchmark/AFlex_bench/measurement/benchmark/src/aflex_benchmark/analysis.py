from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

def aggregate(results:Path):
    rows=[]
    status_counts=defaultdict(int)
    for p in results.glob("*/summary.json"):
        r=json.loads(p.read_text())
        status = r.get("status", "unknown")
        status_counts[status] += 1
        if status=="complete": rows.append(r)
    groups=defaultdict(list)
    for r in rows:
        p=r["point"]; groups[(p.get("rq"),p.get("architecture"),p.get("parallelism"),p.get("nodes"))].append(r)
    agg=[]
    for key,vals in sorted(groups.items(),key=str):
        agg.append({"rq":key[0],"architecture":key[1],"parallelism":key[2],"nodes":key[3],"runs":len(vals),"mean_energy_per_token_j":sum((v.get("energy_per_output_token_j") or 0) for v in vals)/len(vals),"mean_throughput_tokens_s":sum(v.get("throughput_tokens_s",0) for v in vals)/len(vals)})
    blocked = sum(status_counts[status]
                  for status in ("blocked", "requires_preflight"))
    return {"rq5": [x for x in agg if x["rq"]=="RQ5"],
            "rq6":[x for x in agg if x["rq"]=="RQ6"],
            "runs":len(rows), "blocked": blocked,
            "status_counts": dict(sorted(status_counts.items()))}
