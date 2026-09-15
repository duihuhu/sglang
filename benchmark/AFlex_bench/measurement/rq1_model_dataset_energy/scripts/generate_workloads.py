#!/usr/bin/env python3
import hashlib, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(Path(__file__).parent))
from rq1lib import QPS_ORDER, atomic_json, load

def main():
    config=load(ROOT/'configs/workloads.json'); out=ROOT/'data/workloads'; out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for name,spec in config['specs'].items():
        for qps in QPS_ORDER:
            rng=random.Random(config['seed']+spec['input_len']*1009+spec['output_len']*9176+qps)
            current=0.0; rows=[]
            for index in range(config['request_count']):
                current += rng.expovariate(qps)
                rows.append({'request_id':f'rq1-{name}-qps{qps}-{index:05d}',
                    'arrival_time_s':round(current,6),'input_len':spec['input_len'],
                    'output_len':spec['output_len'],'source':f'rq1_fixed_{name}'})
            path=out/f'{name}_qps{qps}.jsonl'
            content=''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows)
            path.write_text(content)
            entries.append({'name':name,'qps':qps,'path':path.name,'requests':len(rows),
                'input_len':spec['input_len'],'output_len':spec['output_len'],
                'seed':config['seed'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    atomic_json(out/'index.json',{'schema_version':1,'frozen':True,'entries':entries})
    print(json.dumps({'files':len(entries),'requests_per_file':config['request_count'],'index':str(out/'index.json')},indent=2))
if __name__=='__main__': main()
