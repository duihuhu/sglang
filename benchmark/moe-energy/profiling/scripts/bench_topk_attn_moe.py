#!/usr/bin/env python3
"""Run real attention once, then profile only the downstream MoE/FFN stage."""
from __future__ import annotations
import argparse,json,multiprocessing,sys,time
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT/'python'),str(HERE)]
from profile_utils import DEFAULT_MODEL,NvmlController,build_forward_batch,load_runner,make_reqs,profile_world,routing_summary,stable_seed
CASES=[('prefill',512,1),('prefill',2048,1),('prefill',4096,1),('decode',64,512),('decode',64,2048),('decode',64,4096)]
def worker(sa,pa,a,gpu,rank):
 from sglang.srt.layers.moe import initialize_moe_config
 from sglang.srt.layers.dp_attention import set_is_extend_in_batch
 from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
 initialize_moe_config(sa);runner=load_runner(sa,pa,gpu,rank);ctrl=NvmlController(gpu);layer=runner.model.model.layers[a.layer_id];f=Path(a.output).open('a',buffering=1) if rank==0 else None
 try:
  ctrl.lock(ctrl.snap(930));time.sleep(.1)
  for phase,length,batch in CASES:
   set_is_extend_in_batch(phase=='prefill');runner.req_to_token_pool.clear();runner.token_to_kv_pool_allocator.clear();reqs=make_reqs(batch,length if phase=='prefill' else length-1,np.random.default_rng(stable_seed(42,phase,length,batch,'req')));fb=build_forward_batch(reqs,runner,phase);n=int(fb.seq_lens_sum) if phase=='prefill' else batch
   # Real model-derived layer input: embeddings, then all preceding full decoder layers.
   with torch.no_grad(),forward_context(ForwardContext(attn_backend=runner.attn_backend)):
    hidden=runner.model.model.embed_tokens(fb.input_ids);residual=None
    for prev in runner.model.model.layers[:a.layer_id]:hidden,residual=prev(fb.positions,hidden,fb,residual)
    attn_in,residual=layer.layer_communicator.prepare_attn_and_capture_last_layer_outputs(hidden,residual,fb)
    attn_out=layer.self_attn(positions=fb.positions,hidden_states=attn_in,forward_batch=fb) if attn_in.shape[0] else attn_in
    # Attention and preceding layers are now complete and synchronized outside profiling.
    torch.cuda.synchronize()
    moe_input,residual_after_attn=layer.layer_communicator.prepare_mlp(attn_out,residual,fb);torch.cuda.synchronize();routing=routing_summary(layer.mlp,moe_input,8)
    def target():return layer.mlp(moe_input,fb)
    lat,rl,re,ene,actual=profile_world(target,ctrl,10,50,8,phase,3)
   if rank==0:
    row={'status':'ok','router_top_k':a.router_top_k,'phase':phase,'layer_id':a.layer_id,'length':length,'batch':batch,'M':n,'input_source':'real_attention_output','profile_segment':'moe_mlp_only_after_prepare_mlp','freq_mhz':930,'latency_us':lat,'latency_per_rank_us':rl,'energy_total_mj':ene,'energy_per_rank_mj':re,'energy_per_token_mj':ene/n,'total_assignments':n*a.router_top_k,'actual_repeat':actual,'routing':routing};f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
 finally:ctrl.close();f and f.close()
def main():
 from sglang.srt.entrypoints.engine import _set_envs_and_config
 from sglang.srt.server_args import PortArgs,ServerArgs
 from sglang.srt.utils import maybe_reindex_device_id
 p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);p.set_defaults(model_path=DEFAULT_MODEL,tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',cuda_graph_backend_decode='disabled',cuda_graph_backend_prefill='disabled',disable_custom_all_reduce=True,max_total_tokens=600000,max_running_requests=4096);p.add_argument('--router-top-k',type=int,required=True);p.add_argument('--layer-id',type=int,default=1);p.add_argument('--output',required=True);a=p.parse_args();sa=ServerArgs.from_cli_args(a);_set_envs_and_config(sa);pa=PortArgs.init_new(sa);ps=[]
 for r in range(8):
  with maybe_reindex_device_id(r) as gpu:x=multiprocessing.Process(target=worker,args=(sa,pa,a,gpu,r));x.start();ps.append(x)
 for x in ps:x.join()
 failed=[(x.pid,x.exitcode) for x in ps if x.exitcode]
 if failed:raise SystemExit(failed)
if __name__=='__main__':main()
