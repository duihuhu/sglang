#!/usr/bin/env python3
"""Schedule mixed MoE EP+TP full-FFN profiling on node3/node4."""
from __future__ import annotations
import argparse, re, shlex, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROFILE_ROOT=HERE.parent
REPO_ROOT=HERE.parents[3]
RAW_ROOT=PROFILE_ROOT/'data'/'mix-EP-TP'/'raw'
OUT_ROOT=PROFILE_ROOT/'data'/'mix-EP-TP'
RAW_SUBDIR='benchmark/moe-energy/profiling/data/mix-EP-TP/raw'
from run_distributed_profile_matrix import DEFAULT_NODES, Job, Node, NodeState, Scheduler, parse_nodes

TOPOLOGIES=((4,2),(8,2),(8,4))
ROUTINGS=('balanced','skewed_rank0')
PHASES=('prefill','decode')
FULL_LENGTHS=(64,256,512,1024)
FULL_BATCHES=(1,8,32,1024)
FULL_FREQS=(210,690,930,1410)
BASIC_PREFILL_LENGTHS=(64,512,4096)
BASIC_DECODE_BATCHES=(64,512,4096)

def stem(phase,routing,tp,ep): return f'qwen3_{phase}_mix_ep{ep}_tp{tp}_{routing}'
def topology_from_stem(s):
    m=re.search(r'_ep(\d+)_tp(\d+)_',s)
    if not m: raise ValueError(s)
    return int(m.group(2)),int(m.group(1))

def make_jobs(args):
    jobs=[]
    for phase in args.phases:
      for routing in args.routing_modes:
       for tp,ep in TOPOLOGIES:
        if args.basic:
          lengths=BASIC_PREFILL_LENGTHS if phase=='prefill' else (64,)
          batches=(1,) if phase=='prefill' else BASIC_DECODE_BATCHES
          expected=len(lengths)*len(batches)
        else:
          lengths=FULL_LENGTHS; batches=FULL_BATCHES
          expected=sum(l*b<=600000 for l in lengths for b in batches)*len(FULL_FREQS)
        jobs.append(Job(phase=phase,label='F-MIX',component='F',mode='moe_ep',size=tp,
                        stem=stem(phase,routing,tp,ep),routing_mode=routing,
                        lengths=tuple(lengths),batch_sizes=tuple(batches),expected_rows=expected))
    return jobs

class MixScheduler(Scheduler):
    def benchmark_command(self,job,output,port):
      tp,ep=topology_from_stem(job.stem)
      cmd=['python3',f'benchmark/moe-energy/profiling/scripts/bench_{job.phase}_af.py',
           '--model-path',self.args.model_path,'--component','F','--parallel-mode','moe_ep',
           '--tp-size',str(tp),'--ep-size',str(ep),'--moe-runner-backend','triton',
           '--moe-a2a-backend','none','--cuda-graph-backend-decode','disabled',
           '--cuda-graph-backend-prefill','disabled','--nccl-port',str(port),
           '--forced-routing',job.routing_mode,'--output',output,'--local-world-size',str(tp),
           '--lengths',*[str(x) for x in job.lengths],
           '--batch-sizes',*[str(x) for x in job.batch_sizes],
           '--freqs',*([str(self.args.basic_freq)] if self.args.basic else [str(x) for x in FULL_FREQS])]
      cmd.extend(shlex.split(self.args.extra_args)); return cmd
    def export_compat(self):
      cmd=[sys.executable,str(HERE/'export_mix_ep_tp.py'),'--raw-root',str(self.raw_root),'--output-dir',str(OUT_ROOT)]
      r=subprocess.run(cmd,text=True,capture_output=True)
      if r.stdout.strip(): print(r.stdout.strip(),flush=True)
      if r.returncode: print(r.stderr,file=sys.stderr)

def dry_run(args,nodes,jobs):
    states={n.name:NodeState(n) for n in nodes}; pending=sorted(jobs,key=lambda j:-j.size); wave=1
    helper=MixScheduler(args,nodes,[])
    while pending:
      assigned=[]
      for job in list(pending):
       for st in sorted(states.values(),key=lambda s:sum(s.used)):
        g=st.allocate(job.size)
        if g is not None:
         out=Path(args.container_output_root)/st.node.name/f'{job.stem}.jsonl'
         print(f'DRY-RUN wave={wave} {job.job_id} node={st.node.name} gpus={g}\n  '+shlex.join(helper.benchmark_command(job,str(out),args.base_nccl_port+g[0])))
         assigned.append((st,g));pending.remove(job);break
      if not assigned: raise SystemExit('cannot fit jobs')
      for st,g in assigned: st.release(g)
      wave+=1

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',nargs='?',choices=['run','status'],default='run')
 p.add_argument('--basic',action='store_true');p.add_argument('--basic-freq',type=int,default=930)
 p.add_argument('--model-path',default='/models/Qwen3-30B-A3B');p.add_argument('--container',default='moe-energy')
 p.add_argument('--container-repo',default='/workspace/sglang-source/sglang');p.add_argument('--host-repo',type=Path,default=REPO_ROOT)
 p.add_argument('--raw-root',type=Path,default=RAW_ROOT);p.add_argument('--container-output-root',default=RAW_SUBDIR)
 p.add_argument('--nodes',nargs='+',default=['node3','node4']);p.add_argument('--node-host',action='append',default=[])
 p.add_argument('--phases',nargs='+',choices=PHASES,default=list(PHASES));p.add_argument('--routing-modes',nargs='+',choices=ROUTINGS,default=list(ROUTINGS))
 p.add_argument('--extra-args',default='--shape-token-limit 0 --max-total-tokens 600000 --warmup 10 --repeat 50 --max-running-requests 4096')
 p.add_argument('--base-nccl-port',type=int,default=30000);p.add_argument('--poll-interval',type=float,default=1.0);p.add_argument('--control-timeout',type=float,default=30.0);p.add_argument('--terminate-timeout',type=float,default=10.0)
 p.add_argument('--retry-failed',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--max-job-attempts',type=int,default=2);p.add_argument('--skip-existing',action='store_true');p.add_argument('--skip-existing-min-rows',type=int,default=1)
 p.add_argument('--dry-run',action='store_true');p.add_argument('--no-export-compat',action='store_true');p.add_argument('--prefer-node1-progress',action=argparse.BooleanOptionalAction,default=False)
 p.add_argument('--components',nargs='+',default=['F-MIX']);p.add_argument('--ep-a2a-backend',default='none');p.add_argument('--ep-max-dispatch-tokens',type=int,default=131072);p.add_argument('--shard-matrix',action='store_true')
 return p.parse_args()
def main():
 args=parse_args()
 if not args.node_host: args.node_host=[f'node3={DEFAULT_NODES["node3"]}',f'node4={DEFAULT_NODES["node4"]}']
 nodes=parse_nodes(args);jobs=make_jobs(args)
 if args.dry_run: dry_run(args,nodes,jobs);return 0
 s=MixScheduler(args,nodes,jobs);rc=s.run()
 if not args.no_export_compat:s.export_compat()
 return rc
if __name__=='__main__':raise SystemExit(main())
