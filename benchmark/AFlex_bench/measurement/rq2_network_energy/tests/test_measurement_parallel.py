import importlib.util, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("parallel",ROOT/"scripts/run_measurement_rdma_parallel.py"); parallel=importlib.util.module_from_spec(spec); spec.loader.exec_module(parallel)
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import expand_queue
class MeasurementParallelTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.bundle=load_bundle(ROOT/"configs",ROOT/"configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json"); cls.base=expand_queue(cls.bundle)[0]
 def test_lanes_exclude_node2_gpu4_and_are_disjoint(self):
  seen={}
  for lane,value in parallel.LANES.items():
   self.assertNotIn(4,value["gpus"]["node2"])
   for node,gpus in value["gpus"].items():
    for gpu in gpus:
     self.assertNotIn((node,gpu),seen); seen[node,gpu]=lane
 def test_configured_jobs_have_unique_ports_and_run_ids(self):
  jobs=[parallel.configure(self.base,1,i,lane,"batch",["a","b","c"])[0] for i,lane in enumerate(("lane0","lane1","lane3"),1)]
  self.assertEqual(len({j["point"]["port_base"] for j in jobs}),3); self.assertEqual(len({j["run_id"] for j in jobs}),3)
 def test_same_batch_has_disjoint_gpu_and_nic_leases(self):
  leases=[]
  for i,lane in enumerate(("lane0","lane1","lane3"),1):
   item,_=parallel.configure(self.base,1,i,lane,"batch",[]); layout=item["point"]["metadata"]["actual_layout"]
   leases.extend((node,gpu,row["nic"]) for node,row in layout.items() for gpu in row["gpus"])
  self.assertEqual(len({(n,g) for n,g,_ in leases}),len(leases)); self.assertEqual(len({(n,nic) for n,_,nic in leases}),6)
 def test_energy_uuid_scope_rejects_node2_gpu4(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/"energy.json").write_text(json.dumps({"gpu_uuids":{"node1":{"0":"GPU-a"},"node2":{"4":"GPU-b"}}}))
   self.assertFalse(parallel.energy_scope_valid(p,{"node1":{"gpus":[0]},"node2":{"gpus":[4]}}))
if __name__=="__main__": unittest.main()
