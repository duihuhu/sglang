#!/usr/bin/env python3
import json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 groups=defaultdict(list)
 for p in (ROOT/"results/raw").rglob("summary.json"):
  try:s=json.loads(p.read_text())
  except Exception:continue
  if s.get("status")!="complete":continue
  manifest=p.parent/"deployment_manifest.json"
  try:m=json.loads(manifest.read_text())
  except Exception:m={}
  meta=m.get("metadata",{}); key=(meta.get("topology_id","unknown"),m.get("architecture",meta.get("architecture","unknown")),s.get("workload","unknown"))
  e=s.get("energy",{}).get("total_j"); q=s.get("achieved_qps")
  if isinstance(e,(int,float)) and isinstance(q,(int,float)):groups[key].append((e,q))
 out=[]
 for key,vals in sorted(groups.items()):out.append({"topology":key[0],"architecture":key[1],"workload":key[2],"repeats":len(vals),"energy_j_mean":statistics.fmean(x[0] for x in vals),"achieved_qps_mean":statistics.fmean(x[1] for x in vals)})
 target=ROOT/"analysis/energy_summary.json";target.write_text(json.dumps({"rq":"RQ2","groups":out},indent=2)+"\n");print(target)
if __name__=="__main__":main()
