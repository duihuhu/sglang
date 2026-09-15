import importlib.util,json,sys,tempfile,unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
spec=importlib.util.spec_from_file_location("r2c",ROOT/"scripts/run_measurement_rdma2n_qps2.py");r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
class ConcurrencyTests(unittest.TestCase):
 def test_singleton_lock(self):
  with tempfile.TemporaryDirectory() as d:
   a=r2.acquire(Path(d),"canary")
   with self.assertRaisesRegex(RuntimeError,"singleton lock held"):r2.acquire(Path(d),"formal")
   a.close()
 def test_cleanup_is_token_scoped(self):
  b=r2.load_bundle_2n();j=r2.build_inventory(b)["jobs"][0];from aflex_benchmark.runner import expand_queue
  base=next(x for x in expand_queue(b) if x["point"]["architecture"]==j["architecture"] and x["workload"]==j["workload"]);item,_=r2.configure(base,j["repeat"],j["ordinal"],j["pair"],j["lane"],[],1);plan=r2.validate_plan_shape(b,item);cmd=" ".join(x[1] for x in plan.cleanup);self.assertIn("AFLEX_RUN_ID",cmd);self.assertNotIn("fuser",cmd);self.assertFalse(plan.unlock)
 def test_signal_flag(self):r2.STOP.set();self.assertTrue(r2.STOP.is_set());r2.STOP.clear()
 def test_max_concurrency_four(self):self.assertEqual(r2.resource_plan()["max_concurrency"],4)
 def _scheduler_fixture(self, root, statuses):
  b=r2.load_bundle_2n();full=r2.build_inventory(b);chosen=[];rows=[]
  for index,status in enumerate(statuses):
   job=dict(full["jobs"][index]);chosen.append(job);rows.append({"repeat":job["repeat"],"architecture":job["architecture"],"workload":job["workload"],"status":status,"attempts":[]});job["logical_id"]=r2.logical_key(job["repeat"],job["architecture"],job["workload"])
  progress={"runs":rows,"valid_runs":sum(x=="valid" for x in statuses),"state":"formal"}
  args=SimpleNamespace(results=Path(root),retry_blocked=True,max_attempts=1,max_concurrency=2,backoff_base=0)
  return args,b,{"jobs":chosen},progress
 def test_future_failure_does_not_cancel_other_job(self):
  with tempfile.TemporaryDirectory() as d:
   args,b,inv,progress=self._scheduler_fixture(d,["pending","pending"]);seen=[]
   def execute(bundle,base,results,job,corunners,attempt,canary=False):
    seen.append(job["logical_id"])
    if job["logical_id"]==inv["jobs"][0]["logical_id"]:raise KeyboardInterrupt("isolated future failure")
    return {"run_id":"ok"},{},{"status":"complete"},Path(d)/"ok",True,None
   with mock.patch.object(r2,"recover_progress",return_value=progress),mock.patch.object(r2,"execute_once",side_effect=execute):result=r2.run_scheduler(args,b,inv)
   self.assertEqual(set(seen),{x["logical_id"] for x in inv["jobs"]});self.assertEqual(result["runs"][1]["status"],"valid")
 def test_exhausted_key_does_not_terminate_queue(self):
  with tempfile.TemporaryDirectory() as d:
   args,b,inv,progress=self._scheduler_fixture(d,["pending","pending"]);args.max_concurrency=1;seen=[]
   def execute(bundle,base,results,job,corunners,attempt,canary=False):
    seen.append(job["logical_id"]);ok=job["logical_id"]==inv["jobs"][1]["logical_id"]
    return {"run_id":job["logical_id"]},{},{"status":"complete" if ok else "failed","failure_stage":"deployment","error":"health timeout"},Path(d)/job["logical_id"],ok,None if ok else "deployment:health timeout"
   with mock.patch.object(r2,"recover_progress",return_value=progress),mock.patch.object(r2,"execute_once",side_effect=execute):result=r2.run_scheduler(args,b,inv)
   self.assertEqual(seen,[x["logical_id"] for x in inv["jobs"]]);self.assertEqual([x["status"] for x in result["runs"]],["blocked","valid"])
 def test_retry_blocked_adds_attempt_allowance(self):
  with tempfile.TemporaryDirectory() as d:
   args,b,inv,progress=self._scheduler_fixture(d,["blocked"]);progress["runs"][0]["attempts"]=[{"attempt":1},{"attempt":2},{"attempt":3}];attempts=[]
   def execute(bundle,base,results,job,corunners,attempt,canary=False):
    attempts.append(attempt);return {"run_id":"ok"},{},{"status":"complete"},Path(d)/"ok",True,None
   with mock.patch.object(r2,"recover_progress",return_value=progress),mock.patch.object(r2,"execute_once",side_effect=execute):result=r2.run_scheduler(args,b,inv)
   self.assertEqual(attempts,[4]);self.assertEqual(result["runs"][0]["status"],"valid")
 def test_failed_summary_reason_precedes_missing_requests(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/"deployment_manifest.json").write_text("{}")
   (root/"summary.json").write_text(json.dumps({"status":"failed","failure_stage":"deployment","error":"D exited"}))
   self.assertEqual(r2.strict_audit(root),(False,"deployment:D exited"))
 def test_same_lane_reservations_conflict_despite_unique_ports(self):
  b=r2.load_bundle_2n();jobs=[x for x in r2.build_inventory(b)["jobs"] if x["lane"]==0][:2]
  first=((jobs[0]["pair"],jobs[0]["lane"],jobs[0]["hca"]),frozenset((x["host"],x["port"]) for x in jobs[0]["ports"]))
  second=((jobs[1]["pair"],jobs[1]["lane"],jobs[1]["hca"]),frozenset((x["host"],x["port"]) for x in jobs[1]["ports"]))
  self.assertEqual(first[0],second[0]);self.assertFalse(first[1]&second[1])
if __name__=="__main__":unittest.main()
