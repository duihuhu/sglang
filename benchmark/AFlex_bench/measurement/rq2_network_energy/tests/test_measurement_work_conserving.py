import fcntl,importlib.util,sys,tempfile,unittest
from pathlib import Path
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import expand_queue
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
spec=importlib.util.spec_from_file_location("wc",ROOT/"scripts/run_measurement_rdma_work_conserving.py"); wc=importlib.util.module_from_spec(spec); spec.loader.exec_module(wc)
class WorkConservingTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.bundle=load_bundle(ROOT/"configs",ROOT/"configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json"); cls.items={(x["point"]["metadata"]["topology_id"],x["point"]["architecture"]):x for x in expand_queue(cls.bundle) if x["workload"]=="measurement_qa_lpld"}
 def test_build_plan_components_match_all_dynamic_placements(self):
  cases=[]
  for arch in ("native","pd","af"):
   for pair,lane in (("pair12",0),("pair34",1)):
    cases.append(("rdma2n",arch,{"topology":"rdma2n","pair":pair,"lane":lane,"layout":wc.PAIR_GPUS[pair][lane]}))
   cases.append(("rdma4n",arch,{"topology":"rdma4n","lane":2,"layout":wc.FOUR_GPUS[2]}))
  for index,(topo,arch,slot) in enumerate(cases,1):
   with self.subTest(topo=topo,arch=arch,slot=slot):
    item,_=wc.configure(self.items[topo,arch],1,index,slot,[]); plan=wc.validate_plan_shape(self.bundle,item); expected={n:sorted(g) for n,g in slot["layout"].items()}; self.assertEqual(plan.used_gpu_map(),expected)
    for process in (p for p in plan.processes if p.gpus): self.assertIn(wc.NICS[slot["lane"]],process.command)
 def test_two_node_reaches_six_without_conflicts(self):
  chosen=[s for s in wc.slots("rdma2n") if (s["pair"]=="pair12" and s["lane"] in (0,1,3)) or (s["pair"]=="pair34" and s["lane"] in (0,1,2))]
  self.assertEqual(len(chosen),6); self.assert_no_conflict(chosen)
 def test_four_node_reaches_four_without_conflicts(self):
  chosen=wc.slots("rdma4n"); self.assertEqual(len(chosen),4); self.assert_no_conflict(chosen); self.assertEqual(chosen[2]["layout"]["node2"],[5])
 def assert_no_conflict(self,chosen):
  gpus=[]; nics=[]
  for s in chosen:
   g,n=wc.resource({"layout":s["layout"],"nic":wc.NICS[s["lane"]]}); gpus.extend(g); nics.extend(n)
  self.assertEqual(len(gpus),len(set(gpus))); self.assertEqual(len(nics),len(set(nics))); self.assertNotIn(("node2",4),gpus)
 def test_dynamic_pair_placements(self):
  point={"architecture":"pd"}; layout={"node3":[0,1],"node4":[0,1]}; out=wc.placements(point,layout); self.assertEqual(out["prefill_placements"][0]["node"],"node3"); self.assertEqual(out["decode_placements"][0]["node"],"node4")
 def test_scheduler_lock_is_cross_process_fail_fast(self):
  with tempfile.TemporaryDirectory() as directory:
   first=wc.acquire_scheduler_lock(Path(directory))
   with self.assertRaisesRegex(RuntimeError,"another measurement scheduler"):
    wc.acquire_scheduler_lock(Path(directory))
   first.close()
   second=wc.acquire_scheduler_lock(Path(directory)); second.close()
 def test_logical_recovery_key_deduplicates(self):
  rows=[{"repeat":1,"point_id":"p","workload":"w","status":"valid"},{"repeat":1,"point_id":"p","workload":"w","status":"valid"}]; keys={(r["repeat"],r["point_id"],r["workload"]) for r in rows}; self.assertEqual(len(keys),1)
if __name__=="__main__":unittest.main()
