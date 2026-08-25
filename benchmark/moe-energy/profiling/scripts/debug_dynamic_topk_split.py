#!/usr/bin/env python3
import argparse,multiprocessing,sys,torch
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT/'python'),str(HERE)]
from profile_utils import DEFAULT_MODEL,load_runner
from bench_dynamic_topk_pilot import full_top8,moe_group_precomputed,shared_importance_mask
def worker(sa,pa,a,gpu,rank):
 from sglang.srt.layers.moe import initialize_moe_config
 from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
 initialize_moe_config(sa);runner=load_runner(sa,pa,gpu,rank);mlp=runner.model.model.layers[1].mlp;mlp.experts.moe_runner_config.inplace=False;item=torch.load(a.inputs/f'decode_M512_rank{rank}.pt');x=item['moe_input'].to(runner.device);importance=item['attention_output_l2'].to(runner.device)
 with torch.no_grad(),forward_context(ForwardContext(attn_backend=runner.attn_backend)):
  logits,w8,ids8=full_top8(mlp,x);ref=moe_group_precomputed(mlp,x,logits,w8,ids8,8);mask,_=shared_importance_mask(importance,.2);imp=mask.nonzero().flatten();normal=(~mask).nonzero().flatten();oi=moe_group_precomputed(mlp,x[imp],logits[imp],w8[imp],ids8[imp],8);on=moe_group_precomputed(mlp,x[normal],logits[normal],w8[normal],ids8[normal],8);split=torch.empty_like(x);split[imp]=oi;split[normal]=on;d=(split.float()-ref.float());print(rank,'xnorm',x.float().norm().item(),'imp',len(imp),'ref',ref.float().norm().item(),'split',split.float().norm().item(),'cos',torch.nn.functional.cosine_similarity(split.float().flatten(),ref.float().flatten(),dim=0).item(),'rel',d.norm().item()/ref.float().norm().item(),'nonzero',torch.count_nonzero(split).item(),flush=True)
def main():
 from sglang.srt.entrypoints.engine import _set_envs_and_config
 from sglang.srt.server_args import PortArgs,ServerArgs
 from sglang.srt.utils import maybe_reindex_device_id
 p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);p.set_defaults(model_path=DEFAULT_MODEL,tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',cuda_graph_backend_decode='disabled',cuda_graph_backend_prefill='disabled',disable_custom_all_reduce=True,max_total_tokens=600000,max_running_requests=4096);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args();sa=ServerArgs.from_cli_args(a);_set_envs_and_config(sa);pa=PortArgs.init_new(sa);ps=[]
 for r in range(8):
  with maybe_reindex_device_id(r) as gpu:q=multiprocessing.Process(target=worker,args=(sa,pa,a,gpu,r));q.start();ps.append(q)
 for q in ps:q.join()
if __name__=='__main__':main()
