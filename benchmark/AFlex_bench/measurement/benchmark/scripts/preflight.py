#!/usr/bin/env python3
import argparse,json,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from aflex_benchmark.config import load_json,validate_cluster
def check(host,container,model_paths):
 inner="python3 --version; nvidia-smi -L; test -d /workspace/moe-tier; "+"; ".join(f"test -e {shlex.quote(p)}" for p in model_paths); argv=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=5",host,f"docker exec {shlex.quote(container)} bash -lc {shlex.quote(inner)}"]; t=time.time()
 try:
  r=subprocess.run(argv,text=True,capture_output=True,timeout=15); return {"ok":r.returncode==0,"returncode":r.returncode,"duration_s":round(time.time()-t,3),"stdout":r.stdout[-4000:],"stderr":r.stderr[-4000:]}
 except Exception as e: return {"ok":False,"error":repr(e),"duration_s":round(time.time()-t,3)}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--output",default=ROOT/"results/preflight.json",type=Path); a=p.parse_args(); cluster=load_json(ROOT/"configs/cluster.json"); models=load_json(ROOT/"configs/models.json")["models"]; validate_cluster(cluster); paths=sorted({m["path"] for m in models.values()}); rows=[]
 for node in cluster["nodes"]: rows.append({"node":node["name"],"host":node["host"],**check(node["host"],cluster["container"],paths)})
 report={"read_only":True,"timestamp_unix_s":time.time(),"checks":rows,"ok":all(x["ok"] for x in rows)}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ok"] else 2
if __name__=="__main__": raise SystemExit(main())
