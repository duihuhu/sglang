from __future__ import annotations
import json,time

def collect_system(executor,cluster,nodes=None):
    rows=[]
    for n in cluster["nodes"][:nodes] if nodes else cluster["nodes"]:
        cmd="nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used --format=csv,noheader,nounits"
        r=executor.run(n["host"],cmd,check=False); rows.append({"node":n["name"],"host":n["host"],"timestamp_unix_s":time.time(),"nvidia_smi_csv":r.stdout or "","returncode":r.returncode})
    return rows
