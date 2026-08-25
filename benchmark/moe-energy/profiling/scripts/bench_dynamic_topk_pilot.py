#!/usr/bin/env python3
"""Two-tier dynamic Top-K pilot on captured real MoE inputs."""
from __future__ import annotations
import argparse,json,multiprocessing,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT/'python'),str(HERE)]
from profile_utils import DEFAULT_MODEL,NvmlController,load_runner,profile_world
CASES=[('prefill',512),('prefill',2048),('prefill',4096),('decode',512),('decode',2048),('decode',4096)]
RATIOS=[.05,.10,.20,.40];LOW_KS=[6,4]
def full_top8(mlp,x):
 from sglang.srt.layers.moe.topk import StandardTopKOutput
 logits,_=mlp.gate(x);probs=torch.softmax(logits.float(),dim=-1);w,ids=probs.topk(8,dim=-1);w=(w/w.sum(-1,keepdim=True)).to(logits.dtype);return logits,w,ids

def moe_group_precomputed(mlp,x,logits,w8,ids8,k):
 from sglang.srt.layers.moe.topk import StandardTopKOutput
 w=w8[:,:k];w=(w/w.sum(-1,keepdim=True)).to(logits.dtype);o=mlp.experts(x,StandardTopKOutput(w,ids8[:,:k],logits))
 from sglang.srt.distributed.communication_op import moe_expert_parallel_all_reduce
 if mlp.ep_size>1:o=moe_expert_parallel_all_reduce(o)
 return o

def shared_importance_mask(score,ratio):
 n=score.shape[0];count=max(1,int(round(n*ratio)));mask=torch.zeros(n,device=score.device,dtype=torch.bool)
 if torch.distributed.get_rank()==0:mask[torch.topk(score,count,sorted=False).indices]=True
 torch.distributed.broadcast(mask,src=0)
 return mask,score

def dyn_forward(mlp,x,logits,w8,ids8,mask,low_k):
 important=mask.nonzero(as_tuple=False).flatten();normal=(~mask).nonzero(as_tuple=False).flatten();out=torch.empty_like(x);out[important]=moe_group_precomputed(mlp,x[important],logits[important],w8[important],ids8[important],8)
 if normal.numel():out[normal]=moe_group_precomputed(mlp,x[normal],logits[normal],w8[normal],ids8[normal],low_k)
 return out
def qm(out,ref,mask):
 d=out.float()-ref.float();rf=ref.float();tok=d.norm(dim=-1)/rf.norm(dim=-1).clamp_min(1e-12)
 def stats(v):
  q=torch.quantile(v,torch.tensor([.5,.95,.99],device=v.device));return {'mean':v.mean().item(),'p50':q[0].item(),'p95':q[1].item(),'p99':q[2].item(),'max':v.max().item()}
 return {'cosine':torch.nn.functional.cosine_similarity(out.float().flatten(),rf.flatten(),dim=0).item(),'nrmse':(d.norm()/rf.norm().clamp_min(1e-12)).item(),'important_token_error':stats(tok[mask]),'normal_token_error':stats(tok[~mask])}
def worker(sa,pa,a,gpu,rank):
 from sglang.srt.layers.moe import initialize_moe_config
 from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
 initialize_moe_config(sa);runner=load_runner(sa,pa,gpu,rank);ctrl=NvmlController(gpu);layer=runner.model.model.layers[1];layer.mlp.experts.moe_runner_config.inplace=False;f=Path(a.output).open('a',buffering=1) if rank==0 else None
 try:
  ctrl.lock(ctrl.snap(930));time.sleep(.1)
  for phase,M in CASES:
   item=torch.load(a.inputs/f'{phase}_M{M}_rank{rank}.pt');x=item['moe_input'].to(runner.device);importance=item['attention_output_l2'].to(runner.device)
   with torch.no_grad(),forward_context(ForwardContext(attn_backend=runner.attn_backend)):
    logits,w8,ids8=full_top8(layer.mlp,x);ref=moe_group_precomputed(layer.mlp,x,logits,w8,ids8,8);torch.cuda.synchronize()
    # Proxy diagnostic: does hidden norm retrieve tokens with largest Top-4 approximation error?
    out4=moe_group_precomputed(layer.mlp,x,logits,w8,ids8,4);err=(out4.float()-ref.float()).norm(dim=-1)/ref.float().norm(dim=-1).clamp_min(1e-12);corr=torch.corrcoef(torch.stack([importance,err]))[0,1].item()
    for ratio in RATIOS:
     mask,shared_score=shared_importance_mask(importance,ratio)
     for low_k in LOW_KS:
      out=dyn_forward(layer.mlp,x,logits,w8,ids8,mask,low_k);quality=qm(out,ref,mask);torch.cuda.synchronize()
      def target():return dyn_forward(layer.mlp,x,logits,w8,ids8,mask,low_k)
      lat,rl,re,ene,actual=profile_world(target,ctrl,5,20,8,phase,3)
      if rank==0:
       row={'status':'ok','phase':phase,'M':M,'importance_proxy':'moe_input_l2_norm','important_ratio':ratio,'important_count':int(mask.sum()),'important_top_k':8,'normal_top_k':low_k,'effective_avg_top_k':ratio*8+(1-ratio)*low_k,'latency_us':lat,'energy_total_mj':ene,'energy_per_token_mj':ene/M,'latency_per_rank_us':rl,'energy_per_rank_mj':re,'quality':quality,'importance_error_pearson':corr,'actual_repeat':actual};f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
 finally:ctrl.close();f and f.close()
def main():
 from sglang.srt.entrypoints.engine import _set_envs_and_config
 from sglang.srt.server_args import PortArgs,ServerArgs
 from sglang.srt.utils import maybe_reindex_device_id
 p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);p.set_defaults(model_path=DEFAULT_MODEL,tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',cuda_graph_backend_decode='disabled',cuda_graph_backend_prefill='disabled',disable_custom_all_reduce=True,max_total_tokens=600000,max_running_requests=4096);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',required=True);a=p.parse_args();sa=ServerArgs.from_cli_args(a);_set_envs_and_config(sa);pa=PortArgs.init_new(sa);ps=[]
 for r in range(8):
  with maybe_reindex_device_id(r) as gpu:x=multiprocessing.Process(target=worker,args=(sa,pa,a,gpu,r));x.start();ps.append(x)
 for x in ps:x.join()
 failed=[(x.pid,x.exitcode) for x in ps if x.exitcode]
 if failed:raise SystemExit(failed)
if __name__=='__main__':main()
