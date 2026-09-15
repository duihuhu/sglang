import subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aflex_benchmark.deploy.base import ProcessSpec,RemoteExecutor
class FakeExecutor(RemoteExecutor):
 def __init__(self,results):super().__init__('x',False,emit=lambda _:None);self.results=list(results);self.commands=[]
 def run(self,host,cmd,**kwargs):
  self.commands.append(cmd);return self.results[0]
def result(rc=0,out=''):return subprocess.CompletedProcess([],rc,out,'')
def spec(node='node1',gpu=0,run='run-good'):
 return ProcessSpec('NATIVE',node,'host',gpu if isinstance(gpu,list) else [gpu],30000,'x',pid_path=f'/tmp/aflex_bench/{run}/{node}_NATIVE_30000.pid',log_path='/tmp/x.log')
class StartupOwnershipTests(unittest.TestCase):
 def test_launcher_and_scheduler_child_pid_different_pass(self):
  e=FakeExecutor([result(0)]);e.verify_gpu_process_binding(spec());cmd=e.commands[0];self.assertIn('AFLEX_RUN_ID=run-good',cmd);self.assertNotIn('/proc/[0-9]*',cmd);self.assertIn('launcher_pgid=$(ps -o pgid= -p $launcher',cmd);self.assertIn("ps -eo pid=,pgid= | awk -v pgid=\"$launcher_pgid\" '$2 == pgid {print $1}'",cmd);self.assertIn('for pid in $group_pids; do proc=/proc/$pid',cmd);self.assertEqual(cmd.count('nvidia-smi -i $gpu --query-compute-apps=pid'),1);self.assertNotIn('test "$pid" = "$launcher"',cmd)
 def test_process_group_scanned_once_while_every_gpu_is_checked(self):
  e=FakeExecutor([result(0)]);e.verify_gpu_process_binding(spec(gpu=[0,2,5]));cmd=e.commands[0]
  self.assertNotIn('/proc/[0-9]*',cmd)
  self.assertEqual(cmd.count('ps -eo pid=,pgid='),1)
  self.assertEqual(cmd.count('for pid in $group_pids'),1)
  self.assertIn('test "$launcher_pgid" = "$launcher"',cmd)
  self.assertEqual(cmd.count('grep -Fxq AFLEX_RUN_ID=run-good'),1)
  self.assertEqual(cmd.count('grep -Fxq AFLEX_COMPONENT_ROLE=NATIVE'),1)
  self.assertEqual(cmd.count('grep -Fxq AFLEX_COMPONENT_PORT=30000'),1)
  self.assertLess(cmd.index('for pid in $group_pids'),cmd.index('gpu=0; uuid='))
  self.assertEqual(cmd.count('nvidia-smi -i $gpu --query-gpu=uuid'),3)
  self.assertEqual(cmd.count('nvidia-smi -i $gpu --query-compute-apps=pid'),3)
  for gpu in (0,2,5):self.assertIn(f'gpu={gpu}; uuid=',cmd)
  for field in ('ancestry=','token_candidates=','owned=','nvml_pid=','pid_namespace_note='):
   self.assertIn(field,cmd)
 def test_external_process_fails(self):
  e=FakeExecutor([result(22,'AFLEX_BARRIER nvml_pid=900 external owned=')]);
  with self.assertRaisesRegex(RuntimeError,'external'):e.verify_gpu_process_binding(spec(),timeout_s=0.01,interval_s=0)
 def test_wrong_run_id_fails(self):
  e=FakeExecutor([result(22,'token_candidates= wrong-run')]);
  with self.assertRaisesRegex(RuntimeError,'wrong-run'):e.verify_gpu_process_binding(spec(),timeout_s=0.01,interval_s=0)
 def test_missing_expected_node_fails(self):
  specs=[spec('node1'),spec('node2')];executors={'node1':FakeExecutor([result(0)]),'node2':FakeExecutor([result(22,'missing expected node2 ownership')])}
  executors['node1'].verify_gpu_process_binding(specs[0])
  with self.assertRaisesRegex(RuntimeError,'missing expected node2'):executors['node2'].verify_gpu_process_binding(specs[1],timeout_s=0.01,interval_s=0)
if __name__=='__main__':unittest.main()
