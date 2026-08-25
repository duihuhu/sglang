#!/usr/bin/env python3
"""Capture real Top-8 attention outputs before the MoE stage for P/D shapes."""
from __future__ import annotations
import argparse,multiprocessing,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(ROOT/'python'),str(HERE)]
from profile_utils import DEFAULT_MODEL,build_forward_batch,dist_barrier,load_runner,make_reqs,stable_seed
CASES=[('prefill',512,1),('prefill',2048,1),('prefill',4096,1),('decode',64,512),('decode',64,2048),('decode',64,4096)]
def worker(sa,pa,a,gpu,rank):
 from sglang.srt.layers.moe import initialize_moe_config
 from sglang.srt.layers.dp_attention import set_is_extend_in_batch
 from sglang.srt.model_executor.forward_context import ForwardContext,forward_context
 initialize_moe_config(sa);runner=load_runner(sa,pa,gpu,rank);layer=runner.model.model.layers[a.layer_id];a.output.mkdir(parents=True,exist_ok=True)
 for phase,length,batch in CASES:
  set_is_extend_in_batch(phase=='prefill');runner.req_to_token_pool.clear();runner.token_to_kv_pool_allocator.clear();reqs=make_reqs(batch,length if phase=='prefill' else length-1,np.random.default_rng(stable_seed(42,phase,length,batch,'req')));fb=build_forward_batch(reqs,runner,phase)
  with torch.no_grad(),forward_context(ForwardContext(attn_backend=runner.attn_backend)):
   hidden=runner.model.model.embed_tokens(fb.input_ids);residual=None
   for prev in runner.model.model.layers[:a.layer_id]:hidden,residual=prev(fb.positions,hidden,fb,residual)
   attn_in,residual=layer.layer_communicator.prepare_attn_and_capture_last_layer_outputs(hidden,residual,fb);attn_out=layer.self_attn(positions=fb.positions,hidden_states=attn_in,forward_batch=fb) if attn_in.shape[0] else attn_in;moe_input,_=layer.layer_communicator.prepare_mlp(attn_out,residual,fb);torch.cuda.synchronize()
  torch.save({'phase':phase,'length':length,'batch':batch,'M':int(fb.seq_lens_sum) if phase=='prefill' else batch,'moe_input':moe_input.cpu(),'attention_output_l2':attn_out.float().norm(dim=-1).cpu()},a.output/f'{phase}_M{int(fb.seq_lens_sum) if phase=="prefill" else batch}_rank{rank}.pt')
  dist_barrier(8)
def main():
 from sglang.srt.entrypoints.engine import _set_envs_and_config
 from sglang.srt.server_args import PortArgs,ServerArgs
 from sglang.srt.utils import maybe_reindex_device_id
 p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);p.set_defaults(model_path=DEFAULT_MODEL,tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',cuda_graph_backend_decode='disabled',cuda_graph_backend_prefill='disabled',disable_custom_all_reduce=True,max_total_tokens=600000,max_running_requests=4096);p.add_argument('--layer-id',type=int,default=1);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sa=ServerArgs.from_cli_args(a);_set_envs_and_config(sa);pa=PortArgs.init_new(sa);ps=[]
 for r in range(8):
  with maybe_reindex_device_id(r) as gpu:x=multiprocessing.Process(target=worker,args=(sa,pa,a,gpu,r));x.start();ps.append(x)
 for x in ps:x.join()
 failed=[(x.pid,x.exitcode) for x in ps if x.exitcode]
 if failed:raise SystemExit(failed)
if __name__=='__main__':main()
