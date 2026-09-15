from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GPURef:
    node: str
    gpu: int

class PortAllocator:
    def __init__(self, start: int=41000, end: int=60999): self.next=start; self.end=end; self.used:set[tuple[str,int]]=set()
    def allocate(self,node:str)->int:
        while (node,self.next) in self.used: self.next+=1
        if self.next>self.end: raise RuntimeError("port range exhausted")
        p=self.next; self.used.add((node,p)); self.next+=1; return p

def select_gpus(cluster:dict,nodes:int)->list[GPURef]:
    chosen=cluster["nodes"][:nodes]
    return [GPURef(n["name"],int(g)) for n in chosen for g in n["gpus"]]

def partition(items:list[GPURef], sizes:list[int])->list[list[GPURef]]:
    if any(s<=0 for s in sizes) or sum(sizes)!=len(items): raise ValueError(f"resource conservation failed: {sum(sizes)} != {len(items)}")
    out=[]; i=0
    for s in sizes: out.append(items[i:i+s]); i+=s
    if len({x for group in out for x in group})!=len(items): raise ValueError("GPU assigned more than once")
    return out
