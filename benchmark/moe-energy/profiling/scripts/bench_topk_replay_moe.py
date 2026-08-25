#!/usr/bin/env python3
"""Replay captured real attention outputs and profile only MoE, with Top-8 output quality."""
from __future__ import annotations
import argparse,json,multiprocessing,sys,time
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT/'python'),str(HERE)]
from profile_utils import DEFAULT_MODEL,NvmlController,build_forward_batch,load_runner,make_reqs,profile_world,routing_summary,stable_seed
CASES=[('prefill',512,1),('prefill',2048,1),('prefill',4096,1),('decode',64,512),('decode',64,2048),('decode',64,4096)]
def metrics(out,ref):
 of=out.float();rf=ref.float();diff=of-rf;tok=(diff.norm(dim=-1)/rf.norm(dim=-1).clamp_min(1e-12));q=torch.quantile(tok,torch.tensor([.5,.9,.95,.99],device=tok.device));return {'cosine':torch.nn.functional.cosine_similarity(of.flatten(),rf.flatten(),dim=0).item(),'nrmse':(diff.norm()/rf.norm().clamp_min(1e-12)).item(),'relative_l1':(diff.abs().sum()/rf.abs().sum().clamp_min(1e-12)).item(),'relative_l2':(diff.norm()/rf.norm().clamp_min(1e-12)).item(),'max_abs_error':diff.abs().max().item(),'token_rel_l2_p50':q[0].item(),'token_rel_l2_p90':q[1].item(),'token_rel_l2_p95':q[2].item(),'token_rel_l2_p99':q[3].item(),'token_rel_l2_max':tok.max().item()}
def worker(sa,pa,a,gpu,rank):
 from sglang.srt.layers.moe import initialize_moe_config
 from sglang.srt.layers.dp_attention import set_is_extend_in_batch
 from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
 initialize_moe_config(sa);runner=load_runner(sa,pa,gpu,rank);ctrl=NvmlController(gpu);layer=runner.model.model.layers[a.layer_id];layer.mlp.experts.moe_runner_config.inplace=False;f=Path(a.output).open('a',buffering=1) if rank==0 else None
 try:
  ctrl.lock(ctrl.snap(930));time.sleep(.1)
  for phase,length,batch in CASES:
   set_is_extend_in_batch(phase=='prefill');item=torch.load(a.inputs/f'{phase}_M{length if phase=="prefill" else batch}_rank{rank}.pt');moe_input=item['moe_input'].to(runner.device);n=item['M']
   with torch.no_grad(),forward_context(ForwardContext(attn_backend=runner.attn_backend)):
    router_logits,_=layer.mlp.gate(moe_input);top8=torch.softmax(router_logits.float(),dim=-1).topk(8,dim=-1).values;dropped=(top8[:,a.router_top_k:].sum(-1)/top8.sum(-1).clamp_min(1e-12)) if a.router_top_k<8 else torch.zeros(n,device=moe_input.device);routing=routing_summary(layer.mlp,moe_input,8);out=layer.mlp(moe_input,None);torch.cuda.synchronize()
    ref_path=a.reference/f'{phase}_M{n}_rank{rank}.pt';quality=None
    if a.router_top_k==8:torch.save(out.cpu(),ref_path)
    else:quality=metrics(out,torch.load(ref_path).to(out.device))
    def target():return layer.mlp(moe_input,None)
    lat,rl,re,ene,actual=profile_world(target,ctrl,10,50,8,phase,3)
   if rank==0:
    ds=dropped.detach().cpu();row={'status':'ok','router_top_k':a.router_top_k,'phase':phase,'M':n,'layer_id':a.layer_id,'input_source':'captured_real_attention_output','profile_segment':'prepare_mlp_done_then_mlp_only','freq_mhz':930,'latency_us':lat,'latency_per_rank_us':rl,'energy_total_mj':ene,'energy_per_rank_mj':re,'energy_per_token_mj':ene/n,'total_assignments':n*a.router_top_k,'dropped_router_mass_mean':ds.mean().item(),'dropped_router_mass_p95':torch.quantile(ds,.95).item(),'dropped_router_mass_p99':torch.quantile(ds,.99).item(),'quality':quality,'routing':routing};f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
 finally:ctrl.close();f and f.close()
def main():
 from sglang.srt.entrypoints.engine import _set_envs_and_config
 from sglang.srt.server_args import PortArgs,ServerArgs
 from sglang.srt.utils import maybe_reindex_device_id
 p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);p.set_defaults(model_path=DEFAULT_MODEL,tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',cuda_graph_backend_decode='disabled',cuda_graph_backend_prefill='disabled',disable_custom_all_reduce=True,max_total_tokens=600000,max_running_requests=4096);p.add_argument('--router-top-k',type=int,required=True);p.add_argument('--layer-id',type=int,default=1);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',required=True);a=p.parse_args();a.reference.mkdir(parents=True,exist_ok=True);sa=ServerArgs.from_cli_args(a);_set_envs_and_config(sa);pa=PortArgs.init_new(sa);ps=[]
 for r in range(8):
  with maybe_reindex_device_id(r) as gpu:x=multiprocessing.Process(target=worker,args=(sa,pa,a,gpu,r));x.start();ps.append(x)
 for x in ps:x.join()
 failed=[(x.pid,x.exitcode) for x in ps if x.exitcode]
 if failed:raise SystemExit(failed)
if __name__=='__main__':main()
