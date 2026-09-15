#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from aflex_benchmark.analysis import aggregate
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import expand_queue
def main():
 p=argparse.ArgumentParser(); p.add_argument("--results",default=ROOT/"results",type=Path); p.add_argument("--output",type=Path); p.add_argument("--matrix",type=Path); p.add_argument("--smoke",action="store_true"); p.add_argument("--allow-experimental",action="store_true"); a=p.parse_args(); report=aggregate(a.results)
 if a.matrix:
  bundle=load_bundle(ROOT/"configs",a.matrix); queue=expand_queue(bundle,smoke=a.smoke,allow_experimental=a.allow_experimental); report["queue"]={"runs":len(queue),"ready":sum(x["state"]=="ready" for x in queue),"blocked":sum(x["state"]!="ready" for x in queue),"requires_preflight":sum(x["state"]=="requires_preflight" for x in queue)}
 text=json.dumps(report,indent=2)+"\n"; a.output.write_text(text) if a.output else print(text,end="")
if __name__=="__main__": main()
