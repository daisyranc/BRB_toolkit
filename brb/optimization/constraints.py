"""Parameter vector views and frozen-entry-safe simplex projection."""
import numpy as np
import torch

class TensorLayout:
    def __init__(self,layout,device,dtype):
        self.layout=layout; self.blocks=[]
        for b in layout.blocks:
            self.blocks.append((b,torch.as_tensor(b.values.copy(),device=device,dtype=dtype),torch.as_tensor(b.trainable,device=device)))

    def unpack(self,vector):
        values={}; offset=0
        for b,base,mask in self.blocks:
            n=int(b.trainable.sum()); values[b.name]=base.masked_scatter(mask,vector[offset:offset+n]); offset+=n
        return values

    def pack(self,values):
        return torch.cat([values[b.name][mask] for b,_,mask in self.blocks]) if self.blocks else torch.empty(0)

    def inventory(self):
        return [{'name':b.name,'shape':list(b.values.shape),'free':int(b.trainable.sum()),'total':b.values.size,'initial':base.detach().cpu().tolist(),'lower':b.lower.tolist(),'upper':b.upper.tolist(),'trainable':b.trainable.tolist()} for b,base,_ in self.blocks]

    def simplex(self,vector,complete_by_name):
        values=self.unpack(vector)
        for b,_,mask in self.blocks:
            if b.name not in complete_by_name: continue
            value=values[b.name].clone(); complete=complete_by_name[b.name]
            lo=torch.as_tensor(b.lower,device=value.device,dtype=value.dtype); hi=torch.as_tensor(b.upper,device=value.device,dtype=value.dtype)
            active=mask.any(1)&(torch.as_tensor(complete,device=value.device)|(value.sum(1)>1))
            target=1-torch.where(mask,torch.zeros_like(value),value).sum(1)
            minimum=torch.where(mask,lo,torch.zeros_like(lo)).sum(1);maximum=torch.where(mask,hi,torch.zeros_like(hi)).sum(1)
            if torch.any(active&((target<minimum-1e-9)|(target>maximum+1e-9))): raise ValueError('Frozen beliefs/bounds make row normalization infeasible.')
            left=torch.where(mask,value-hi,torch.full_like(value,float('inf'))).min(1).values-1
            right=torch.where(mask,value-lo,torch.full_like(value,float('-inf'))).max(1).values+1
            left=torch.where(active,left,torch.zeros_like(left));right=torch.where(active,right,torch.ones_like(right))
            for _ in range(55):
                middle=(left+right)/2
                projected=(value-middle[:,None]).clamp(lo,hi)
                greater=torch.where(mask,projected,torch.zeros_like(projected)).sum(1)>target
                left=torch.where(greater,middle,left);right=torch.where(greater,right,middle)
            projected=(value-((left+right)/2)[:,None]).clamp(lo,hi)
            value=torch.where(active[:,None]&mask,projected,value)
            values[b.name]=value
        return self.pack(values)
