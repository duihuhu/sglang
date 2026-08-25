#!/usr/bin/env python3
"""Capture real Qwen3 routed expert IDs across decode iterations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sglang import Engine
from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info
PROMPTS=[
 "Explain why the sky is blue.","Solve 17 times 23 step by step.","Write a short Python binary search.",
 "Summarize the causes of World War I.","Explain photosynthesis.","Describe how TCP reliability works.",
 "What makes a number prime?","Explain inflation simply.","Compare RAM and SSD storage.",
 "Why does ice float?","Describe recursion with an example.","Explain the role of kidneys."
]
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);x=a.parse_args();e=Engine(model_path='/models/Qwen3-30B-A3B',tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',disable_cuda_graph=True,disable_custom_all_reduce=True,max_total_tokens=32768,max_running_requests=64,enable_return_routed_experts=True)
 try:
  outs=e.generate(prompt=PROMPTS,sampling_params={'temperature':0,'max_new_tokens':32,'ignore_eos':True},return_routed_experts=True)
  samples=[]
  for i,o in enumerate(outs):
   arr=extract_routed_experts_from_meta_info(o).reshape(-1,48,8);samples.append({'request':i,'prompt_tokens':o['meta_info']['prompt_tokens'],'completion_tokens':o['meta_info']['completion_tokens'],'experts':arr.tolist()})
  x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps({'ep_size':8,'experts_per_rank':16,'num_layers':48,'top_k':8,'samples':samples},separators=(',',':')));print(x.output)
 finally:
  try:e.shutdown()
  except Exception:pass
if __name__=='__main__':main()
