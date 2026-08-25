#!/usr/bin/env python3
"""Capture 300 official GSM8K requests' token×layer×topk routing in chunks."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sglang import Engine
from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--num-requests',type=int,default=300);a.add_argument('--chunk-size',type=int,default=50);x=a.parse_args();x.output.mkdir(parents=True,exist_ok=True);ds=load_dataset('openai/gsm8k','main',split='test');indices=list(range(min(x.num_requests,len(ds))));orders={'original':indices,'shuffled':random.Random(42).sample(indices,len(indices))};e=Engine(model_path='/models/Qwen3-30B-A3B',tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',disable_cuda_graph=True,disable_custom_all_reduce=True,max_total_tokens=65536,max_running_requests=64,enable_return_routed_experts=True)
 meta=(x.output/'requests.jsonl').open('w',buffering=1);summary={'dataset':'openai/gsm8k','split':'test','ep_size':8,'experts_per_rank':16,'num_layers':48,'top_k':8,'num_requests_per_order':len(indices),'orders':list(orders)}
 try:
  for order_name,order in orders.items():
   for chunk_id,start in enumerate(range(0,len(order),x.chunk_size)):
    ids=order[start:start+x.chunk_size];prompts=[f"Question: {ds[i]['question']}\nAnswer:" for i in ids];outs=e.generate(prompt=prompts,sampling_params={'temperature':0,'max_new_tokens':32,'ignore_eos':True},return_routed_experts=True)
    arrays={}
    for j,(idx,o) in enumerate(zip(ids,outs)):
     arr=extract_routed_experts_from_meta_info(o).reshape(-1,48,8);key=f'{order_name}_{start+j:04d}';arrays[key]=arr.astype(np.int16);meta.write(json.dumps({'key':key,'order':order_name,'submit_index':start+j,'dataset_index':idx,'prompt_tokens':o['meta_info']['prompt_tokens'],'completion_tokens':o['meta_info']['completion_tokens'],'total_route_rows':len(arr),'question':ds[idx]['question']},separators=(',',':'))+'\n')
    np.savez_compressed(x.output/f'{order_name}_chunk_{chunk_id:03d}.npz',**arrays);print('CAPTURED',order_name,chunk_id,len(ids),flush=True)
  (x.output/'manifest.json').write_text(json.dumps(summary,indent=2))
 finally:
  meta.close()
  try:e.shutdown()
  except Exception:pass
if __name__=='__main__':main()
