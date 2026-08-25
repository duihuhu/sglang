#!/usr/bin/env python3
"""Model-level static Router Top-K quality screen against Top-8 continuations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
from sglang import Engine
PROMPTS=[
 "The capital of France is", "Explain why the sky appears blue.", "What is 17 multiplied by 23?",
 "Write a Python function that reverses a list.", "A train travels 60 miles in 90 minutes. Its average speed is",
 "The process of photosynthesis converts", "Summarize the causes of World War I.", "If all mammals are warm-blooded and whales are mammals, then",
 "Translate to French: Good morning, how are you?", "What is the derivative of x squared plus 3x?",
 "Name three properties of prime numbers.", "Complete the analogy: bird is to nest as bee is to",
 "Why does ice float on water?", "Write a SQL query to count users by country.",
 "The opposite of scarce is", "A rectangle has length 8 and width 5. Its area is",
 "Explain the difference between RAM and storage.", "What does the HTTP 404 status code mean?",
 "In one sentence, define machine learning.", "Solve: 3x + 7 = 25.",
 "Which planet is known as the Red Planet?", "Describe one benefit and one risk of nuclear energy.",
 "What is the time complexity of binary search?", "Continue the sequence: 2, 4, 8, 16,",
 "A fair coin is tossed twice. The probability of two heads is", "What gas do humans breathe in for respiration?",
 "Rewrite politely: Send me the report now.", "What is the main function of the kidneys?",
 "Give a concise explanation of recursion.", "If today is Monday, what day is it in 10 days?",
 "What is the chemical symbol for gold?", "Explain why regular backups are important."
]
def engine():
 return Engine(model_path='/models/Qwen3-30B-A3B',tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',disable_cuda_graph=True,disable_custom_all_reduce=True,max_total_tokens=32768,max_running_requests=64)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--top-k',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);a=ap.parse_args();tok=AutoTokenizer.from_pretrained('/models/Qwen3-30B-A3B',trust_remote_code=True);e=engine()
 try:
  if a.top_k==8 and not a.baseline.exists():
   outs=e.generate(prompt=PROMPTS,sampling_params={'temperature':0,'max_new_tokens':16},return_logprob=True,top_logprobs_num=20)
   base=[]
   for prompt,o in zip(PROMPTS,outs):base.append({'prompt':prompt,'prompt_ids':tok.encode(prompt),'output_ids':o['output_ids'],'text':o['text']})
   a.baseline.parent.mkdir(parents=True,exist_ok=True);a.baseline.write_text(json.dumps(base,indent=2))
  base=json.loads(a.baseline.read_text());inputs=[x['prompt_ids']+x['output_ids'] for x in base];starts=[len(x['prompt_ids']) for x in base]
  outs=e.generate(input_ids=inputs,sampling_params={'temperature':0,'max_new_tokens':1},return_logprob=True,logprob_start_len=starts,top_logprobs_num=20)
  raw=[]
  for item,o,start in zip(base,outs,starts):
   mi=o['meta_info'];raw.append({'prompt':item['prompt'],'reference_output_ids':item['output_ids'],'input_token_logprobs':mi.get('input_token_logprobs',[]),'input_top_logprobs':mi.get('input_top_logprobs',[]),'output_id':o['output_ids'][0] if o['output_ids'] else None})
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'router_top_k':a.top_k,'samples':raw},indent=2));print(a.output)
 finally:e.shutdown()
if __name__=='__main__':main()
