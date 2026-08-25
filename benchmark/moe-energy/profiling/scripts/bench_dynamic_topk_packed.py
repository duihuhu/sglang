#!/usr/bin/env python3
"""Single-MoE-call dynamic Top-K with physically packed ragged assignments."""
from __future__ import annotations
import argparse,json,multiprocessing,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT/'python'),str(HERE)]
from profile_utils import DEFAULT_MODEL,NvmlController,load_runner,profile_world
CASES=[('prefill',512),('prefill',2048),('prefill',4096),('decode',512),('decode',2048),('decode',4096)];RATIOS=[.05,.10,.20,.40];LOW_KS=[6,4]
def full_top8(mlp,x):
 logits,_=mlp.gate(x);probs=torch.softmax(logits.float(),dim=-1);w,ids=probs.topk(8,dim=-1);w=(w/w.sum(-1,keepdim=True)).to(logits.dtype);return logits,w,ids
def shared_mask(score,ratio):
 n=score.shape[0];count=max(1,int(round(n*ratio)));mask=torch.zeros(n,device=score.device,dtype=torch.bool)
 if torch.distributed.get_rank()==0:mask[torch.topk(score,count,sorted=False).indices]=True
 torch.distributed.broadcast(mask,src=0);return mask
def packed_forward(mlp,x,logits,w8,ids8,mask,low_k):
 from sglang.srt.layers.moe.topk import StandardTopKOutput
 from sglang.srt.distributed.communication_op import moe_expert_parallel_all_reduce
 per_k=torch.where(mask,torch.full_like(mask,8,dtype=torch.int64),torch.full_like(mask,low_k,dtype=torch.int64));slots=torch.arange(8,device=x.device).unsqueeze(0);valid=slots<per_k.unsqueeze(1);token=torch.arange(x.shape[0],device=x.device).unsqueeze(1).expand_as(ids8)[valid];fx=x[token];fid=ids8[valid].reshape(-1,1);fw=w8[valid].reshape(-1,1);den=(w8*valid).sum(-1);fw=(fw/den[token].unsqueeze(1).clamp_min(1e-12)).to(logits.dtype);flogits=logits[token];fo=mlp.experts(fx,StandardTopKOutput(fw,fid,flogits));out=torch.zeros_like(x);out.index_add_(0,token,fo)
 if mlp.ep_size>1:out=moe_expert_parallel_all_reduce(out)
 return out
def qm(out,ref,mask):
 d=out.float()-ref.float();rf=ref.float();tok=d.norm(dim=-1)/rf.norm(dim=-1).clamp_min(1e-12)
 def st(v):
  q=torch.quantile(v,torch.tensor([.5,.95,.99],device=v.device));return {'mean':v.mean().item(),'p50':q[0].item(),'p95':q[1].item(),'p99':q[2].item(),'max':v.max().item()}
 return {'cosine':torch.nn.functional.cosine_similarity(out.float().flatten(),rf.flatten(),dim=0).item(),'nrmse':(d.norm()/rf.norm().clamp_min(1e-12)).item(),'important_token_error':st(tok[mask]),'normal_token_error':st(tok[~mask])}
def worker(sa,pa,a,gpu,rank):
 from sglang.srt.layers.moe import initialize_moe_config
 from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
 initialize_moe_config(sa);runner=load_runner(sa,pa,gpu,rank);ctrl=NvmlController(gpu);mlp=runner.model.model.layers[1].mlp;mlp.experts.moe_runner_config.inplace=False;f=Path(a.output).open('a',buffering=1) if rank==0 else None
 try:
  ctrl.lock(ctrl.snap(930));time.sleep(.1)
  for phase,M in CASES:
   item=torch.load(a.inputs/f'{phase}_M{M}_rank{rank}.pt');x=item['moe_input'].to(runner.device);importance=item['attention_output_l2'].to(runner.device)
   with torch.no_grad(),forward_context(ForwardContext(attn_backend=runner.attn_backend)):
    logits,w8,ids8=full_top8(mlp,x);mask_cache={r:shared_mask(importance,r) for r in RATIOS};ref=packed_forward(mlp,x,logits,w8,ids8,torch.ones(M,device=x.device,dtype=torch.bool),8);torch.cuda.synchronize()
    # Correlation diagnostic against static Top-4 token error.
    out4=packed_forward(mlp,x,logits,w8,ids8,torch.zeros(M,device=x.device,dtype=torch.bool),4);err=(out4.float()-ref.float()).norm(dim=-1)/ref.float().norm(dim=-1).clamp_min(1e-12);corr=torch.corrcoef(torch.stack([importance,err]))[0,1].item()
    for ratio in RATIOS:
     mask=mask_cache[ratio]
     for low_k in LOW_KS:
      out=packed_forward(mlp,x,logits,w8,ids8,mask,low_k);quality=qm(out,ref,mask);torch.cuda.synchronize()
      def target():
       target_logits,target_w8,target_ids8=full_top8(mlp,x)
       return packed_forward(mlp,x,target_logits,target_w8,target_ids8,mask,low_k)
      lat,rl,re,ene,actual=profile_world(target,ctrl,5,20,8,phase,3)
      if rank==0:
       row={'status':'ok','implementation':'onepass_packed_ragged_assignments','phase':phase,'M':M,'importance_proxy':'pre_norm_attention_output_l2','important_ratio':ratio,'important_top_k':8,'normal_top_k':low_k,'effective_avg_top_k':ratio*8+(1-ratio)*low_k,'physical_assignment_count':int(mask.sum())*8+int((~mask).sum())*low_k,'latency_us':lat,'energy_total_mj':ene,'energy_per_token_mj':ene/M,'latency_per_rank_us':rl,'energy_per_rank_mj':re,'quality':quality,'importance_error_pearson':corr,'actual_repeat':actual};f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
 finally:ctrl.close();f and f.close()
def main():
 from sglang.srt.entrypoints.engine import _set_envs_and_config
 from sglang.srt.server_args import PortArgs,ServerArgs
 from sglang.srt.utils import maybe_reindex_device_id
 p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);p.set_defaults(model_path=DEFAULT_MODEL,tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',cuda_graph_backend_decode='disabled',cuda_graph_backend_prefill='disabled',disable_custom_all_reduce=True,max_total_tokens=600000,max_running_requests=4096);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',required=True);a=p.parse_args();sa=ServerArgs.from_cli_args(a);_set_envs_and_config(sa);pa=PortArgs.init_new(sa);ps=[]
 for r in range(8):
  with maybe_reindex_device_id(r) as gpu:q=multiprocessing.Process(target=worker,args=(sa,pa,a,gpu,r));q.start();ps.append(q)
 for q in ps:q.join()
 failed=[(q.pid,q.exitcode) for q in ps if q.exitcode]
 if failed:raise SystemExit(failed)
if __name__=='__main__':main()
