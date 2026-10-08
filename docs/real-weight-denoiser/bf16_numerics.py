"""Bounded BF16 diagnostics; original acceptance constants remain unchanged."""
import torch

from bf16_contracts import ATOL, RTOL
from metrics import CHUNK, compare


def inspect_values(actual,reference):
    result=compare(actual,reference,atol=ATOL,rtol=RTOL)
    a=actual.detach().reshape(-1);r=reference.detach().reshape(-1)
    edges=torch.tensor([0.,1e-5,1e-4,1e-3,1e-2,.02,.05,.1,.5,1.,10.,float('inf')],dtype=torch.float64)
    histogram=torch.zeros(len(edges)-1,dtype=torch.int64);points=[];worst=None
    squared=reference_squared=0.
    for start in range(0,a.numel(),CHUNK):
        av=a[start:start+CHUNK].to(device='cpu',dtype=torch.float64)
        rv=r[start:start+CHUNK].to(device='cpu',dtype=torch.float64)
        finite=torch.isfinite(av)&torch.isfinite(rv);error=(av-rv).abs();bound=ATOL+RTOL*rv.abs()
        squared+=float(error[finite].square().sum());reference_squared+=float(rv[finite].square().sum())
        histogram+=torch.histogram(error[finite],bins=edges).hist.to(torch.int64)
        def point(i):
            return dict(flat_index=start+i,actual=float(av[i]),reference=float(rv[i]),error=float(error[i]),bound=float(bound[i]),ratio=float(error[i]/bound[i]))
        if bool(finite.any()):
            ratio=torch.where(finite,error/bound,torch.full_like(error,-1));i=int(ratio.argmax());candidate=point(i)
            if worst is None or candidate['ratio']>worst['ratio']:worst=candidate
        for i in ((error>bound)&finite).nonzero().flatten()[:max(0,32-len(points))]:points.append(point(int(i)))
    finite_count=int(histogram.sum());quantiles={}
    for q in (.5,.9,.99,1.):
        if finite_count:
            index=int(torch.searchsorted(histogram.cumsum(0),torch.tensor(max(1,q*finite_count))).clamp(max=len(histogram)-1))
            quantiles[str(q)]=[float(edges[index]),None if not torch.isfinite(edges[index+1]) else float(edges[index+1])]
    result.update(max_normalized_point=worst,failing_points=points,coordinates_truncated=result['violations']>len(points),
        absolute_error_quantile_bins=quantiles,quantile_note='fixed histogram intervals, finite elements only',
        squared_error=squared,reference_squared=reference_squared,squared_sums_scope='finite-elements-only',
        finite_count=finite_count,error_histogram_counts=histogram.tolist(),
        error_histogram_edges=[float(value) if torch.isfinite(value) else None for value in edges],shape=list(actual.shape))
    return result


def expected_head_ownership(full,rank,world):
    width=full.shape[2]//world
    return full[:,:,rank*width:(rank+1)*width].contiguous()


def advance_pair(reference,parallel,reference_step,parallel_step):
    return reference_step(reference),parallel_step(parallel)
