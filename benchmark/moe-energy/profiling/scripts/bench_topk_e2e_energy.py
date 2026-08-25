#!/usr/bin/env python3
"""End-to-end batched generation latency and NVML cluster energy by Router Top-K."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import pynvml
from sglang import Engine
PROMPTS=[f"User request {i}: Explain a useful fact about science, mathematics, history, or computing in a concise paragraph.\nAssistant:" for i in range(32)]
def main():
 a=argparse.ArgumentParser();a.add_argument('--top-k',type=int,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args();pynvml.nvmlInit();hs=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(8)];e=Engine(model_path='/models/Qwen3-30B-A3B',tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',disable_cuda_graph=True,disable_custom_all_reduce=True,max_total_tokens=32768,max_running_requests=64)
 try:
  e.generate(prompt=PROMPTS[:2],sampling_params={'temperature':0,'max_new_tokens':8});before=[pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in hs];t=time.perf_counter();outs=e.generate(prompt=PROMPTS,sampling_params={'temperature':0,'max_new_tokens':64,'ignore_eos':True});lat=time.perf_counter()-t;after=[pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in hs];tokens=sum(o['meta_info']['completion_tokens'] for o in outs);r={'router_top_k':x.top_k,'requests':len(outs),'output_tokens':tokens,'wall_latency_s':lat,'throughput_tok_s':tokens/lat,'cluster_energy_j':sum(b-a for a,b in zip(before,after))/1000,'energy_per_output_token_j':sum(b-a for a,b in zip(before,after))/1000/tokens};x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(r,indent=2));print(r)
 finally:
  try:e.shutdown()
  except Exception:pass
  pynvml.nvmlShutdown()
if __name__=='__main__':main()
