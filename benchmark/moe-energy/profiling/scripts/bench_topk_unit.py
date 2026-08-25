#!/usr/bin/env python3
import sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'python'))
from sglang.srt.layers.moe.topk import TopK

def main():
 torch.manual_seed(7);hidden=torch.randn(32,64,device='cuda',dtype=torch.bfloat16);logits=torch.randn(32,128,device='cuda',dtype=torch.float32)
 outputs={}
 for k in (1,2,4,6,8):
  op=TopK(top_k=k,renormalize=True,use_grouped_topk=False,layer_id=0).cuda();o=op.forward_native(hidden,logits);ids=o.topk_ids;weights=o.topk_weights
  assert ids.shape==(32,k) and weights.shape==(32,k)
  assert torch.allclose(weights.sum(-1),torch.ones(32,device='cuda'),atol=2e-6)
  assert all(torch.unique(row).numel()==k for row in ids)
  outputs[k]=(ids,weights)
 ids8,w8=outputs[8];ref_ids=torch.topk(torch.softmax(logits,dim=-1),8,dim=-1).indices
 assert torch.equal(ids8,ref_ids)
 print('PASS top-k shapes, uniqueness, renormalization, native Top-8 equivalence')
if __name__=='__main__':main()
