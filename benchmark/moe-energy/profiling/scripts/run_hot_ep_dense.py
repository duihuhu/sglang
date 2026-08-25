#!/usr/bin/env python3
"""Run EP4 hot-rank-fraction full-FFN pilot on node3/node4."""
from __future__ import annotations
import argparse,shlex,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
from run_distributed_profile_matrix import DEFAULT_NODES,Job,Scheduler,parse_nodes
MODES=('hot_rank_03125','hot_rank_0375','hot_rank_04375','hot_rank_05625','hot_rank_0625','hot_rank_06875','hot_rank_08125','hot_rank_09375')
class HotScheduler(Scheduler):
 def benchmark_command(self,j,out,port):
  script=f'benchmark/moe-energy/profiling/scripts/bench_{j.phase}_af.py'
  lengths=('512','1024','4096') if j.phase=='prefill' else ('64',)
  batches=('1',) if j.phase=='prefill' else ('512','1024','4096')
  return ['python3',script,'--model-path',self.args.model_path,'--component','F','--parallel-mode','moe_ep','--tp-size','4','--ep-size','4','--moe-runner-backend','triton','--moe-a2a-backend','none','--cuda-graph-backend-decode','disabled','--cuda-graph-backend-prefill','disabled','--disable-custom-all-reduce','--nccl-port',str(port),'--forced-routing',j.routing_mode,'--output',out,'--local-world-size','4','--lengths',*lengths,'--batch-sizes',*batches,'--freqs','930','--shape-token-limit','0','--max-total-tokens','600000','--warmup','10','--repeat','50','--max-running-requests','4096']
 def export_compat(self):
  r=subprocess.run([sys.executable,str(HERE/'export_hot_ep.py'),'--raw-root',str(self.raw_root),'--output-dir',str(HERE.parent/'data'/'hot-EP')],text=True,capture_output=True)
  if r.stdout.strip():print(r.stdout.strip(),flush=True)
def args():
 p=argparse.ArgumentParser();p.add_argument('--model-path',default='/models/Qwen3-30B-A3B');p.add_argument('--container',default='moe-energy');p.add_argument('--container-repo',default='/workspace/sglang-source/sglang');p.add_argument('--host-repo',type=Path,default=ROOT);p.add_argument('--raw-root',type=Path,default=HERE.parent/'data'/'hot-EP'/'raw-dense');p.add_argument('--container-output-root',default='benchmark/moe-energy/profiling/data/hot-EP/raw-dense');p.add_argument('--nodes',nargs='+',default=['node3','node4']);p.add_argument('--node-host',action='append',default=[]);p.add_argument('--base-nccl-port',type=int,default=30200);p.add_argument('--poll-interval',type=float,default=1.0);p.add_argument('--control-timeout',type=float,default=30.0);p.add_argument('--terminate-timeout',type=float,default=10.0);p.add_argument('--retry-failed',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--max-job-attempts',type=int,default=2);p.add_argument('--skip-existing',action='store_true');p.add_argument('--skip-existing-min-rows',type=int,default=1);p.add_argument('--no-export-compat',action='store_true');p.add_argument('--prefer-node1-progress',action=argparse.BooleanOptionalAction,default=False);p.add_argument('--components',nargs='+',default=['F-HOT']);p.add_argument('--ep-a2a-backend',default='none');p.add_argument('--ep-max-dispatch-tokens',type=int,default=131072);p.add_argument('--shard-matrix',action='store_true');return p.parse_args()
def main():
 a=args();a.node_host=[f'node3={DEFAULT_NODES["node3"]}',f'node4={DEFAULT_NODES["node4"]}'];nodes=parse_nodes(a)
 jobs=[Job(phase=ph,label='F-HOT',component='F',mode='moe_ep',size=4,stem=f'qwen3_{ph}_ep4_{m}',routing_mode=m,expected_rows=3) for ph in ('prefill','decode') for m in MODES]
 s=HotScheduler(a,nodes,jobs);rc=s.run();return rc
if __name__=='__main__':raise SystemExit(main())
