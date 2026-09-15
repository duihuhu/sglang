import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
spec=importlib.util.spec_from_file_location("r2",ROOT/"scripts/run_measurement_rdma2n_qps2.py");r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
class RDMA2NTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.bundle=r2.load_bundle_2n();cls.inv=r2.build_inventory(cls.bundle)
 def test_inventory_54_unique(self):self.assertEqual(len(self.inv["jobs"]),54);self.assertEqual(len({x["logical_id"] for x in self.inv["jobs"]}),54)
 def test_only_nodes34_four_lanes(self):self.assertEqual(self.inv["resource_plan"]["planned_nodes"],["node3","node4"]);self.assertEqual(set(r2.LANES),{0,2,4,6})
 def test_three_arch_placements(self):
  for a in r2.ARCHITECTURES:
   j=next(x for x in self.inv["jobs"] if x["architecture"]==a);self.assertEqual(set(j["actual_layout"]),{"node3","node4"});self.assertEqual(sum(len(v["gpus"]) for v in j["actual_layout"].values()),2)
 def test_run_identity_repeat_attempt_slot(self):
  from aflex_benchmark.runner import expand_queue
  base=next(x for x in expand_queue(self.bundle) if x["point"]["architecture"]=="native");a,_=r2.configure(base,1,1,"pair34",0,[],1);b,_=r2.configure(base,2,1,"pair34",0,[],1);c,_=r2.configure(base,1,1,"pair34",0,[],2);self.assertEqual(len({a["run_id"],b["run_id"],c["run_id"]}),3)
 def test_ports_unique_across_lanes(self):
  chosen=[next(x for x in self.inv["jobs"] if x["lane"]==l) for l in r2.LANES]
  sets=[{(p["host"],p["port"]) for p in x["ports"]} for x in chosen]
  for i in range(4):
   for j in range(i):self.assertFalse(sets[i]&sets[j])
 def test_same_logical_key_attempt_ports_are_disjoint(self):
  job=self.inv["jobs"][0];sets=[{(p["host"],p["port"]) for p in job["port_leases"][str(a)]} for a in range(1,r2.FORMAL_PORT_ATTEMPTS+1)]
  for i,current in enumerate(sets):
   for previous in sets[:i]:self.assertFalse(current&previous)
 def test_all_jobs_all_attempt_ports_are_globally_disjoint(self):
  owner={}
  for job in self.inv["jobs"]:
   for attempt,ports in job["port_leases"].items():
    for lease in ((p["host"],p["port"]) for p in ports):
     self.assertNotIn(lease,owner,msg=f"{lease} shared by {owner.get(lease)} and {(job['logical_id'],attempt)}")
     owner[lease]=(job["logical_id"],attempt)
 def test_allocator_bounds_and_avoids_legacy_22100(self):
  bases={r2.allocate_port_base(o,a) for o in range(1,55) for a in range(1,r2.FORMAL_PORT_ATTEMPTS+1)}
  self.assertEqual(len(bases),54*r2.FORMAL_PORT_ATTEMPTS);self.assertGreaterEqual(min(bases),24000);self.assertLessEqual(max(bases)+r2.DERIVED_PORT_MAX_OFFSET,65535)
  all_ports={p["port"] for job in self.inv["jobs"] for ports in job["port_leases"].values() for p in ports};self.assertNotIn(22100,all_ports)
 def test_run_identity_and_gate_hash_do_not_depend_on_allocator_lease(self):
  from aflex_benchmark.runner import expand_queue
  base=next(x for x in expand_queue(self.bundle) if x["point"]["architecture"]=="af");item,_=r2.configure(base,1,3,"pair34",2,[],2)
  self.assertEqual(item["point"]["metadata"]["run_identity"]["attempt"],2);self.assertEqual(r2.hashes(self.bundle,1)["config_hash"],r2.hashes(self.bundle,4)["config_hash"])
 def test_atomic_crash_recovery(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"p.json";r2.atomic_write(p,{"a":1});(Path(d)/".p.json.bad.tmp").write_text("{");self.assertEqual(json.loads(p.read_text()),{"a":1})
 def test_gate_bindings_fail_closed(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(RuntimeError,"gate missing"):r2.require_gate(Path(d),self.bundle)
 def test_gate_binding_is_independent_of_runtime_concurrency(self):
  self.assertEqual(r2.require_gate(r2.ROOT/"results/raw/measurement_rdma2n_qps2_20260825",self.bundle,2)["status"],"pass")
 def test_failure_classification(self):self.assertEqual(r2.classify_failure("health timeout"),"retryable");self.assertEqual(r2.classify_failure("placement invalid"),"non_retryable")
 def test_paused_resume_preserves_valid(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=r2.recover_progress(root,self.inv,True);p["runs"][0]["status"]="valid";p["runs"][0]["attempts"]=[{"valid":True}];r2.atomic_write(root/"progress.json",p);q=r2.recover_progress(root,self.inv,True);self.assertEqual(q["state"],"paused");self.assertEqual(q["valid_runs"],1)
 def test_offline_never_ssh(self):
  with tempfile.TemporaryDirectory() as d:self.assertFalse(r2.offline_preflight(Path(d))["ssh_performed"])
if __name__=="__main__":unittest.main()
