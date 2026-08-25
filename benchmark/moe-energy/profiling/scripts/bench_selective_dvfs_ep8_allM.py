#!/usr/bin/env python3
"""Selective per-rank DVFS pilot: active ranks 930 MHz, inactive ranks lower."""
from __future__ import annotations
import argparse,json,multiprocessing,sys,time
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT/'python'),str(HERE)]
from profile_utils import DEFAULT_MODEL,NvmlController,build_forward_batch,forced_routing,load_runner,make_reqs,profile_world,stable_seed
CASES=[(512,1),(768,2),(1024,2),(1280,3),(1536,4),(1792,5),(2048,8),(2560,7),(3072,8),(3584,7),(4096,8)];FREQS=[930,690,450,210]
def worker(sa,pa,a,gpu,rank):
 from sglang.srt.layers.moe import initialize_moe_config
 from sglang.srt.layers.dp_attention import set_is_extend_in_batch
 from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
 initialize_moe_config(sa);runner=load_runner(sa,pa,gpu,rank);set_is_extend_in_batch(False);ctrl=NvmlController(gpu);layer=runner.model.model.layers[0];order=list(FREQS);np.random.default_rng(a.order_seed).shuffle(order);f=Path(a.output).open('a',buffering=1) if rank==0 else None
 try:
  for M,K in CASES:
   runner.req_to_token_pool.clear();runner.token_to_kv_pool_allocator.clear();reqs=make_reqs(M,63,np.random.default_rng(stable_seed(42,'decode',64,M,'req')));fb=build_forward_batch(reqs,runner,'decode');hidden=torch.randn(M,runner.model_config.hidden_size,device=runner.device,dtype=torch.bfloat16,generator=torch.Generator(device=runner.device).manual_seed(stable_seed(42,'decode',64,M,'hidden')));residual=hidden.clone();layer.mlp._forced_active_rank_count=K;old=forced_routing(layer.mlp,'active_ranks_equal',8)
   for inactive_freq in order:
    target_freq=930 if rank<K else inactive_freq;ctrl.lock(ctrl.snap(target_freq));time.sleep(.1)
    with torch.no_grad(),forward_context(ForwardContext(attn_backend=runner.attn_backend)):
     def target():x,_=layer.post_attention_layernorm(hidden,residual);return layer.mlp(x,fb)
     lat,rl,re,ene,actual=profile_world(target,ctrl,10,50,8,'decode',3)
    if rank==0:
     row={'status':'ok','M':M,'active_rank_count':K,'active_freq_mhz':930,'inactive_freq_mhz':inactive_freq,'latency_us':lat,'latency_per_rank_us':rl,'energy_total_mj':ene,'energy_per_rank_mj':re,'actual_repeat':actual,'replicate':a.replicate,'node':a.node,'order_seed':a.order_seed};f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
   layer.mlp.topk.forward=old
 finally:ctrl.close();f and f.close()
def main():
 from sglang.srt.entrypoints.engine import _set_envs_and_config
 from sglang.srt.server_args import PortArgs,ServerArgs
 from sglang.srt.utils import maybe_reindex_device_id
 p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);p.set_defaults(model_path=DEFAULT_MODEL,tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',cuda_graph_backend_decode='disabled',cuda_graph_backend_prefill='disabled',disable_custom_all_reduce=True,max_total_tokens=600000,max_running_requests=4096);p.add_argument('--output',required=True);p.add_argument('--replicate',type=int,required=True);p.add_argument('--node',required=True);p.add_argument('--order-seed',type=int,required=True);a=p.parse_args();sa=ServerArgs.from_cli_args(a);_set_envs_and_config(sa);pa=PortArgs.init_new(sa);ps=[]
 for r in range(8):
  with maybe_reindex_device_id(r) as gpu:x=multiprocessing.Process(target=worker,args=(sa,pa,a,gpu,r));x.start();ps.append(x)
 for x in ps:x.join()
 failed=[(x.pid,x.exitcode) for x in ps if x.exitcode]
 if failed:raise SystemExit(failed)
if __name__=='__main__':main()
