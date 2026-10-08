"""Frozen review-only FP32 candidate v1; not used by production inference."""
import torch

CANDIDATE_ID='fp32-partial512-fp64-outputsum-v1'
CHUNK=512


class ControlledProjection:
    """FP32 local-row GEMMs, bounded K, FP64 output accumulation, one cast.

    Packs transposed weight chunks once. This adds one weight-sized allocation;
    packing cost and bytes must be reported separately. No row padding/duplication.
    """
    def __init__(self,weight):
        if weight.dtype!=torch.float32 or weight.ndim!=2:
            raise ValueError('Candidate v1 requires a two-dimensional FP32 weight')
        self.inputs=weight.shape[1];self.outputs=weight.shape[0]
        self.parts=tuple(weight[:,start:start+CHUNK].t().contiguous() for start in range(0,self.inputs,CHUNK))
        self.packed_bytes=sum(part.numel()*part.element_size() for part in self.parts)

    def __call__(self,value):
        if value.dtype!=torch.float32 or value.shape[-1]!=self.inputs:
            raise ValueError('Candidate v1 requires matching FP32 inputs')
        total=torch.zeros((*value.shape[:-1],self.outputs),dtype=torch.float64,device=value.device)
        for index,weight in enumerate(self.parts):
            partial=value[...,index*CHUNK:index*CHUNK+weight.shape[0]].contiguous()@weight
            total.add_(partial.double())
        return total.float()
