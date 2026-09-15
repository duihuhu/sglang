#!/usr/bin/env python3
import argparse,json,statistics,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import run_measurement_rdma2n_qps2 as r2
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--results",type=Path,default=ROOT/"results/raw/measurement_rdma2n_qps2_20260825");a=p.parse_args(argv);progress=json.loads((a.results/"progress.json").read_text());groups=defaultdict(list);artifacts=defaultdict(list)
 for row in progress.get("runs",[]):
  if row.get("status")!="valid":continue
  attempt=next((x for x in reversed(row.get("attempts",[])) if x.get("valid")),None)
  if not attempt:continue
  path=r2.resolve_artifact_path(attempt["artifact"]);ok,_=r2.strict_audit(path,actual=attempt.get("layout"))
  if not ok:continue
  s=json.loads((path/"summary.json").read_text());key=(row["architecture"],row["workload"]);groups[key].append({"achieved_qps":s["achieved_qps"],"energy_j":s["energy"]["total_j"]});artifacts[key].append(str(path))
 complete=progress.get("state")=="complete" and progress.get("valid_runs")==54 and len(groups)==18 and all(len(x)==3 for x in groups.values());report={"schema_version":2,"complete":complete,"valid_runs":sum(map(len,groups.values())),"expected_valid_runs":54,"groups":[]}
 for key,rows in sorted(groups.items()):report["groups"].append({"architecture":key[0],"workload":key[1],"repeats":len(rows),"artifacts":artifacts[key],"metrics":{n:{"mean":statistics.fmean(x[n] for x in rows),"sample_std":statistics.stdev(x[n] for x in rows) if len(rows)>1 else None} for n in rows[0]}})
 out=a.results/("measurement_rdma2n_report.json" if complete else "measurement_rdma2n_interim.json");r2.atomic_write(out,report);print(json.dumps({"complete":complete,"valid":report["valid_runs"],"output":str(out)},indent=2));return 0 if complete else 3
if __name__=="__main__":raise SystemExit(main())
