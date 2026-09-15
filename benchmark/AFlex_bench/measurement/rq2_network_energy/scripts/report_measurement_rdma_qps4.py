#!/usr/bin/env python3
from __future__ import annotations
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results/raw/measurement_rdma_qps4_20260825"
def values(summary,link):
 rdma=sum(float(row.get("counter_value") or 0) for row in link.get("paths",[]) if row.get("path")=="rdma")
 return {"success_rate":summary["requests_success"]/summary["requests_total"],"achieved_qps":summary["achieved_qps"],"input_throughput_tokens_s":summary["input_throughput_tokens_s"],"output_throughput_tokens_s":summary["output_throughput_tokens_s"],"ttft_p90_ms":summary["metrics"]["ttft_client_ms"]["p90"],"tpot_p90_ms":summary["metrics"]["tpot_ms"]["p90"],"e2e_p90_ms":summary["metrics"]["e2e_ms"]["p90"],"total_energy_j":summary["energy"]["total_j"],"energy_per_output_token_j":summary["energy_per_output_token_j"],"average_power_w":summary["average_cluster_power_w"],"rdma_bytes":rdma}
def main():
 progress=json.loads((RESULTS/"progress.json").read_text()); groups=defaultdict(list); artifacts=defaultdict(list)
 for row in progress["runs"]:
  if row["status"]!="valid": continue
  attempt=next(a for a in reversed(row["attempts"]) if a["valid"]); path=Path(attempt["artifact"]); summary=json.loads((path/"summary.json").read_text()); link=json.loads((path/"link_validation.json").read_text()); point=summary["point"]; key=(point["metadata"]["topology_id"],point["architecture"],summary["workload"]); groups[key].append(values(summary,link)); artifacts[key].append(str(path))
 progress_metadata={key:progress.get(key) for key in ("parallel_execution","resource_topology","isolation_policy")}
 report={"schema_version":1,"std_definition":"sample standard deviation (n-1)","complete":len(groups)==36 and all(len(rows)==3 for rows in groups.values()),"execution":progress_metadata,"groups":[]}
 flat=[]
 for key,rows in sorted(groups.items()):
  metrics={name:{"mean":statistics.fmean(r[name] for r in rows),"sample_std":statistics.stdev(r[name] for r in rows)} for name in rows[0]}
  item={"topology":key[0],"architecture":key[1],"workload":key[2],"repeats":len(rows),"metrics":metrics,"artifacts":artifacts[key],"anomalies":[]}; report["groups"].append(item)
  flat.append({"topology":key[0],"architecture":key[1],"workload":key[2],"repeats":len(rows),**{f"{m}_{s}":v for m,x in metrics.items() for s,v in x.items()}})
 (RESULTS/"measurement_report.json").write_text(json.dumps(report,indent=2)+"\n")
 with (RESULTS/"measurement_report.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
 lines=["# Measurement RQ2 RDMA QPS4 report","",f"Complete: {report['complete']}","",f"Parallel execution: {report['execution'].get('parallel_execution')}","",f"Isolation policy: `{json.dumps(report['execution'].get('isolation_policy'),sort_keys=True)}`","",f"Fabric caveat: {report['execution'].get('isolation_policy',{}).get('fabric_caveat')}","","| Topology | Architecture | Workload | Repeats | QPS mean±std | Energy J mean±std | J/output-token mean±std | RDMA bytes mean±std |","|---|---|---|---:|---:|---:|---:|---:|"]
 for x in report["groups"]:
  m=x["metrics"]; fmt=lambda n:f"{m[n]['mean']:.6g} ± {m[n]['sample_std']:.6g}"; lines.append(f"| {x['topology']} | {x['architecture']} | {x['workload']} | {x['repeats']} | {fmt('achieved_qps')} | {fmt('total_energy_j')} | {fmt('energy_per_output_token_j')} | {fmt('rdma_bytes')} |")
 (RESULTS/"measurement_report.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
