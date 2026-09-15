#!/usr/bin/env python3
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 "pcie_nvlink_2gpu_qps4":"measurement_local_2gpu_pcie_nvlink_20260827",
 "rdma_2gpu_qps2":"measurement_rdma2n_qps2_20260825",
 "rdma_4gpu_2node_qps4":"measurement_rdma2n_4gpu_qps4_20260825",
 "rdma_4gpu_4node_qps4":"measurement_rdma_qps4_20260825",
}
def main():
 rows=[]
 for experiment,directory in SOURCES.items():
  base=ROOT/"results/raw"/directory
  summaries=list(base.rglob("summary.json"))
  complete=0
  for p in summaries:
   try: complete+=json.loads(p.read_text()).get("status")=="complete"
   except Exception: pass
  rows.append({"experiment":experiment,"directory":str(base),"summary_files":len(summaries),"complete_summaries":complete})
 out={"schema_version":1,"rq":"RQ2","experiments":rows,"primary":"rdma_4gpu_*_qps4","supplementary":["pcie_nvlink_2gpu_qps4","rdma_2gpu_qps2"]}
 target=ROOT/"analysis/experiment_index.json";target.write_text(json.dumps(out,indent=2)+"\n");print(target)
if __name__=="__main__":main()
