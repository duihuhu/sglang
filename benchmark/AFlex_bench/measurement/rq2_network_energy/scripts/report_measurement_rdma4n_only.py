#!/usr/bin/env python3
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/"results/raw/measurement_rdma_qps4_20260825"
def main():
 progress=json.loads((RESULTS/"progress.json").read_text());groups=defaultdict(list);arts=defaultdict(list)
 assert progress.get("phase")=="rdma4n_only"
 for row in progress["runs"]:
  if row["status"]!="valid":continue
  attempt=next(a for a in reversed(row["attempts"]) if a["valid"]);path=Path(attempt["artifact"]);summary=json.loads((path/"summary.json").read_text());link=json.loads((path/"link_validation.json").read_text());key=(summary["point"]["architecture"],summary["workload"]);metrics=summary["metrics"]
  groups[key].append({"success_rate":summary["requests_success"]/64,"achieved_qps":summary["achieved_qps"],"input_throughput_tokens_s":summary["input_throughput_tokens_s"],"output_throughput_tokens_s":summary["output_throughput_tokens_s"],"ttft_p90_ms":metrics["ttft_client_ms"]["p90"],"tpot_p90_ms":metrics["tpot_ms"]["p90"],"e2e_p90_ms":metrics["e2e_ms"]["p90"],"total_energy_j":summary["energy"]["total_j"],"energy_per_output_token_j":summary["energy_per_output_token_j"],"average_power_w":summary["average_cluster_power_w"],"rdma_bytes":sum(float(x.get("counter_value") or 0) for x in link.get("paths",[]) if x.get("path")=="rdma")});arts[key].append(str(path))
 report={"schema_version":1,"phase":"rdma4n_only","expected_valid_runs":54,"complete":len(groups)==18 and all(len(x)==3 for x in groups.values()),"parallel_execution":{"gpu_lanes":[0,2,6],"hca_by_lane":{"0":"mlx5_0","2":"mlx5_1","6":"mlx5_5"},"hca_shared":False,"max_concurrency":3},"std_definition":"sample standard deviation n-1","groups":[]};flat=[]
 for (arch,workload),rows in sorted(groups.items()):
  m={k:{"mean":statistics.fmean(x[k] for x in rows),"sample_std":statistics.stdev(x[k] for x in rows)} for k in rows[0]};item={"architecture":arch,"workload":workload,"repeats":len(rows),"metrics":m,"artifacts":arts[arch,workload]};report["groups"].append(item);flat.append({"architecture":arch,"workload":workload,"repeats":len(rows),**{f"{k}_{s}":v for k,x in m.items() for s,v in x.items()}})
 (RESULTS/"measurement_rdma4n_report.json").write_text(json.dumps(report,indent=2)+"\n")
 if flat:
  with (RESULTS/"measurement_rdma4n_report.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 lines=["# RQ2 RDMA4N QPS4 report","",f"Complete: {report['complete']}","","GPU lanes: 0,2,6 on all four nodes, bound respectively to mlx5_0, mlx5_1, and mlx5_5. No HCA is shared between concurrent jobs.",""]
 for x in report["groups"]:lines.append(f"- {x['architecture']} / {x['workload']}: {x['repeats']}/3 valid repeats")
 (RESULTS/"measurement_rdma4n_report.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"complete":report["complete"],"groups":len(report["groups"])},indent=2))
if __name__=="__main__":main()
