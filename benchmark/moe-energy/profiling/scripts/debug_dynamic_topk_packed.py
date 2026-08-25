#!/usr/bin/env python3
"""Validate packed flat assignments against native width-8 MoE."""
from __future__ import annotations
import argparse,multiprocessing,sys,torch
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT/'python'),str(HERE)]
from profile_utils import DEFAULT_MODEL,load_runner
from bench_dynamic_topk_pilot import full_top8,moe_group_precomputed
def packed(mlp,x,logits,w8,ids8,valid):
 from sglang.srt.layers.moe.topk import StandardTopKOutput
 from sglang.srt.distributed.communication_op import moe_expert_parallel_all_reduce
 token=torch.arange(x.shape[0],device=x.device).unsqueeze(1).expand_as(ids8)[valid];fx=x[token];fid=ids8[valid].reshape(-1,1);fw=w8[valid].reshape(-1,1);flogits=logits[token];fo=mlp.experts(fx,StandardTopKOutput(fw,fid,flogits));out=torch.zeros_like(x);out.index_add_(0,token,fo)
 if mlp.ep_size>1:out=moe_expert_parallel_all_reduce(out)
 return out
def worker(sa,pa,a,gpu,rank):
 from sglang.srt.layers.moe import initialize_moe_config
 from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
 initialize_moe_config(sa);runner=load_runner(sa,pa,gpu,rank);mlp=runner.model.model.layers[1].mlp;mlp.experts.moe_runner_config.inplace=False;item=torch.load(a.inputs/f'decode_M512_rank{rank}.pt');x=item['moe_input'].to(runner.device)
 with torch.no_grad(),forward_context(ForwardContext(attn_backend=runner.attn_backend)):
  logits,w8,ids8=full_top8(mlp,x);ref=moe_group_precomputed(mlp,x,logits,w8,ids8,8);o=packed(mlp,x,logits,w8,ids8,torch.ones_like(ids8,dtype=torch.bool));d=o.float()-ref.float();print(rank,'cos',torch.nn.functional.cosine_similarity(o.float().flatten(),ref.float().flatten(),dim=0).item(),'rel',d.norm().item()/ref.float().norm().item(),'refnorm',ref.float().norm().item(),'onorm',o.float().norm().item(),flush=True)
def main():
 from sglang.srt.entrypoints.engine import _set_envs_and_config
 from sglang.srt.server_args import PortArgs,ServerArgs
 from sglang.srt.utils import maybe_reindex_device_id
 p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);p.set_defaults(model_path=DEFAULT_MODEL,tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',cuda_graph_backend_decode='disabled',cuda_graph_backend_prefill='disabled',disable_custom_all_reduce=True,max_total_tokens=600000,max_running_requests=4096);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();sa=ServerArgs.from_cli_args(a);_set_envs_and_config(sa);pa=PortArgs.init_new(sa);ps=[]
 for r in range(8):
  with maybe_reindex_device_id(r) as gpu:q=multiprocessing.Process(target=worker,args=(sa,pa,a,gpu,r));q.start();ps.append(q)
 for q in ps:q.join()
if __name__=='__main__':main()
