#!/usr/bin/env python3
"""Offline reproducible MMLU/GSM8K-style proxy task eval for Router Top-K."""
from __future__ import annotations
import argparse,json,re,random,statistics
from pathlib import Path
from sglang import Engine
MCQ=[
("What is the capital of France?",["Berlin","Madrid","Paris","Rome"],"C"),("Which planet is the Red Planet?",["Earth","Mars","Venus","Jupiter"],"B"),("Water freezes at what Celsius temperature?",["0","10","32","100"],"A"),("Binary search has what time complexity?",["O(1)","O(log n)","O(n)","O(n^2)"],"B"),("HTTP 404 means?",["Success","Unauthorized","Not Found","Server Error"],"C"),("Chemical symbol for gold?",["Ag","Au","Fe","Gd"],"B"),("Largest ocean?",["Atlantic","Indian","Pacific","Arctic"],"C"),("Who wrote Hamlet?",["Shakespeare","Dickens","Homer","Austen"],"A"),("Derivative of x^2?",["x","2x","x^2","2"],"B"),("Prime among these?",["21","27","29","33"],"C"),("Photosynthesis mainly uses?",["Oxygen and glucose","CO2 and water","Nitrogen and salt","Hydrogen and iron"],"B"),("RAM is primarily?",["Persistent storage","Volatile memory","A CPU core","A network protocol"],"B"),("The kidneys primarily?",["Pump blood","Filter blood","Digest protein","Control lungs"],"B"),("A mammal is?",["Shark","Whale","Trout","Lizard"],"B"),("SQL COUNT returns?",["Rows count","Column type","Table name","Index size"],"A"),("Opposite of scarce?",["Rare","Abundant","Small","Costly"],"B"),("Probability of two heads in two fair tosses?",["1/2","1/3","1/4","3/4"],"C"),("A triangle angles sum to?",["90","180","270","360"],"B"),("TCP is?",["Transport protocol","Database","Language","File format"],"A"),("Which is renewable?",["Coal","Oil","Solar","Natural gas"],"C"),
("DNA carries?",["Sound","Genetic information","Heat","Electric current"],"B"),("Inflation means generally?",["Falling price level","Rising price level","No trade","Fixed wages"],"B"),("The legislative branch primarily?",["Makes laws","Enforces laws","Interprets laws","Prints money only"],"A"),("A compiler converts?",["Source code","Images","Network packets","SQL rows"],"A"),("IPv4 address length?",["16 bits","32 bits","64 bits","128 bits"],"B"),("Mitochondria produce?",["ATP","DNA only","Cell walls","Chlorophyll"],"A"),("Supply rises, demand fixed: price tends to?",["Rise","Fall","Double always","Become zero"],"B"),("A valid Python list literal?",["{1,2}","[1,2]","(1=>2)","<1,2>"],"B"),("The equator divides Earth into?",["East/West","North/South","Land/Sea","Hot/Cold"],"B"),("Conservation of energy says energy?",["Is created","Is destroyed","Transforms but total conserved","Always decreases"],"C"),("HTTPS adds?",["Compression only","Encryption/authentication","More HTML","A database"],"B"),("A noun names?",["Action only","Person/place/thing","Color only","Punctuation"],"B"),]
# deterministic grade-school arithmetic, answer is an integer
GSM=[]
rng=random.Random(7)
for i in range(48):
 a=rng.randint(10,90);b=rng.randint(2,12);c=rng.randint(1,20)
 typ=i%4
 if typ==0:q=f"A store has {a} boxes with {b} pencils each and sells {c} pencils. How many pencils remain?";ans=a*b-c
 elif typ==1:q=f"A bus travels {a} km in the morning and {b*5} km in the afternoon for {c} days. How many km total?";ans=(a+b*5)*c
 elif typ==2:q=f"There are {a} students split equally into {b} groups, with {a%b} students sitting out. How many students are in each full group?";ans=(a-a%b)//b
 else:q=f"Mia has {a} dollars, earns {b*c} dollars, then spends {c} dollars. How many dollars remain?";ans=a+b*c-c
 GSM.append((q,ans))
def eng():return Engine(model_path='/models/Qwen3-30B-A3B',tp_size=8,ep_size=8,moe_runner_backend='triton',moe_a2a_backend='none',disable_cuda_graph=True,disable_custom_all_reduce=True,max_total_tokens=32768,max_running_requests=64)
def ci(vals,n=2000):
 rr=random.Random(123);means=[statistics.mean(rr.choice(vals) for _ in vals) for _ in range(n)];means.sort();return means[int(.025*n)],means[int(.975*n)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--top-k',type=int,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();e=eng()
 try:
  prompts=[q+'\n'+'\n'.join(f'{chr(65+j)}. {x}' for j,x in enumerate(opts))+'\nAnswer with one letter:' for q,opts,_ in MCQ];outs=e.generate(prompt=prompts,sampling_params={'temperature':0,'max_new_tokens':2});pred=[re.search(r'[ABCD]',o['text'].strip().upper()) for o in outs];mc=[int(bool(m) and m.group(0)==label) for m,(_,_,label) in zip(pred,MCQ)]
  gp=[q+' Give only the final integer answer.' for q,_ in GSM];go=e.generate(prompt=gp,sampling_params={'temperature':0,'max_new_tokens':64});gsm=[];preds=[]
  for o,(_,ans) in zip(go,GSM):
   nums=re.findall(r'-?\d+',o['text'].replace(',',''));v=int(nums[-1]) if nums else None;preds.append(v);gsm.append(int(v==ans))
  result={'router_top_k':a.top_k,'mmlu_proxy_accuracy':statistics.mean(mc),'mmlu_proxy_ci95':ci(mc),'mmlu_proxy_n':len(mc),'gsm8k_proxy_accuracy':statistics.mean(gsm),'gsm8k_proxy_ci95':ci(gsm),'gsm8k_proxy_invalid_rate':statistics.mean(v is None for v in preds),'gsm8k_proxy_n':len(gsm),'mmlu_predictions':[m.group(0) if m else '' for m in pred],'gsm8k_predictions':preds};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(result)
 finally:
  pass
if __name__=='__main__':main()
