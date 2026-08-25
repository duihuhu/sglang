#!/usr/bin/env python3
"""Launch one Top-K server, run MMLU/GSM8K pilots, stop it, repeat."""
from __future__ import annotations
import json,os,signal,subprocess,time
from pathlib import Path
import requests
ROOT=Path('/workspace/sglang-source/sglang');OUT=ROOT/'benchmark/moe-energy/profiling/data/Top-k/raw/eval';OUT.mkdir(parents=True,exist_ok=True)
def wait(port,proc):
 for _ in range(240):
  if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}')
  try:
   if requests.get(f'http://127.0.0.1:{port}/health',timeout=1).status_code==200:return
  except Exception:pass
  time.sleep(1)
 raise TimeoutError(port)
def main():
 for k in (8,6,4):
  port=32000+k;env=os.environ.copy();env.update(SGLANG_PROFILE_ROUTER_TOP_K=str(k),FLASHINFER_DISABLE_VERSION_CHECK='1')
  log=(OUT/f'topk_{k}_server.log').open('w');cmd=['python3','-m','sglang.launch_server','--model-path','/models/Qwen3-30B-A3B','--tp-size','8','--ep-size','8','--moe-runner-backend','triton','--moe-a2a-backend','none','--disable-cuda-graph','--disable-custom-all-reduce','--max-total-tokens','32768','--port',str(port)]
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   wait(port,proc)
   mmlu=['python3','benchmark/mmlu/bench_sglang.py','--host','127.0.0.1','--port',str(port),'--nsub','5','--ntrain','3','--parallel','16','--result-file',str(OUT/f'topk_{k}_mmlu_result.jsonl'),'--raw-result-file',str(OUT/f'topk_{k}_mmlu_raw.json')]
   gsm=['python3','benchmark/gsm8k/bench_sglang.py','--host','127.0.0.1','--port',str(port),'--num-questions','100','--num-shots','5','--max-new-tokens','256','--parallel','16','--result-file',str(OUT/f'topk_{k}_gsm8k_result.jsonl'),'--raw-result-file',str(OUT/f'topk_{k}_gsm8k_raw.json')]
   subprocess.run(mmlu,cwd=ROOT,env=env,check=True,stdout=(OUT/f'topk_{k}_mmlu.log').open('w'),stderr=subprocess.STDOUT)
   subprocess.run(gsm,cwd=ROOT,env=env,check=True,stdout=(OUT/f'topk_{k}_gsm8k.log').open('w'),stderr=subprocess.STDOUT)
   print(f'EVAL_DONE_{k}',flush=True)
  finally:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   log.close();time.sleep(5)
if __name__=='__main__':main()
