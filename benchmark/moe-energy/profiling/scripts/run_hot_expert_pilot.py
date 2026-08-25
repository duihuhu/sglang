#!/usr/bin/env python3
"""Run EP4 within-rank expert-concentration pilot at hot fraction 0.75."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
from run_distributed_profile_matrix import DEFAULT_NODES,Job,Scheduler,parse_nodes
ACTIVE=(32,8,2,1)
class S(Scheduler):
 def benchmark_command(self,j,out,port):
  n=int(j.stem.rsplit('_a',1)[1]);lens=('512','1024','4096') if j.phase=='prefill' else ('64',);batches=('1',) if j.phase=='prefill' else ('512','1024','4096')
  return ['python3',f'benchmark/moe-energy/profiling/scripts/bench_{j.phase}_af.py','--model-path',self.args.model_path,'--component','F','--parallel-mode','moe_ep','--tp-size','4','--ep-size','4','--moe-runner-backend','triton','--moe-a2a-backend','none','--cuda-graph-backend-decode','disabled','--cuda-graph-backend-prefill','disabled','--disable-custom-all-reduce','--nccl-port',str(port),'--forced-routing','hot_rank_0750','--forced-active-experts-per-rank',str(n),'--output',out,'--local-world-size','4','--lengths',*lens,'--batch-sizes',*batches,'--freqs','930','--shape-token-limit','0','--max-total-tokens','600000','--warmup','10','--repeat','50','--max-running-requests','4096']
 def export_compat(self):
  r=subprocess.run([sys.executable,str(HERE/'export_hot_expert.py'),'--raw-root',str(self.raw_root),'--output-dir',str(HERE.parent/'data'/'hot-EP')],text=True,capture_output=True);print(r.stdout.strip(),flush=True)
def args():
 p=argparse.ArgumentParser();p.add_argument('--model-path',default='/models/Qwen3-30B-A3B');p.add_argument('--container',default='moe-energy');p.add_argument('--container-repo',default='/workspace/sglang-source/sglang');p.add_argument('--host-repo',type=Path,default=ROOT);p.add_argument('--raw-root',type=Path,default=HERE.parent/'data'/'hot-EP'/'raw-expert');p.add_argument('--container-output-root',default='benchmark/moe-energy/profiling/data/hot-EP/raw-expert');p.add_argument('--nodes',nargs='+',default=['node3','node4']);p.add_argument('--node-host',action='append',default=[]);p.add_argument('--base-nccl-port',type=int,default=30300);p.add_argument('--poll-interval',type=float,default=1.0);p.add_argument('--control-timeout',type=float,default=30.0);p.add_argument('--terminate-timeout',type=float,default=10.0);p.add_argument('--retry-failed',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--max-job-attempts',type=int,default=2);p.add_argument('--skip-existing',action='store_true');p.add_argument('--skip-existing-min-rows',type=int,default=1);p.add_argument('--no-export-compat',action='store_true');p.add_argument('--prefer-node1-progress',action=argparse.BooleanOptionalAction,default=False);p.add_argument('--components',nargs='+',default=['F-HOT-EXPERT']);p.add_argument('--ep-a2a-backend',default='none');p.add_argument('--ep-max-dispatch-tokens',type=int,default=131072);p.add_argument('--shard-matrix',action='store_true');return p.parse_args()
def main():
 a=args();a.node_host=[f'node3={DEFAULT_NODES["node3"]}',f'node4={DEFAULT_NODES["node4"]}'];nodes=parse_nodes(a);jobs=[Job(ph,'F-HOT-EXPERT','F','moe_ep',4,f'qwen3_{ph}_ep4_hot075_a{n}',routing_mode='hot_rank_0750',expected_rows=3) for ph in ('prefill','decode') for n in ACTIVE];s=S(a,nodes,jobs);rc=s.run();s.export_compat();return rc
if __name__=='__main__':raise SystemExit(main())
