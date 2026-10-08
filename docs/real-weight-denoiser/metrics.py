"""Fixed acceptance with bounded CPU comparison chunks; never infer ULPs."""
import math

import torch

ATOL = .02
RTOL = .02
CHUNK = 262144


def compare(actual, reference, *, atol=ATOL, rtol=RTOL):
    if actual.shape != reference.shape:
        raise ValueError('Comparison shapes differ')
    a, r = actual.detach().reshape(-1), reference.detach().reshape(-1)
    count, bad, nonfinite = a.numel(), 0, 0
    squared, reference_squared, maximum, ratio_max = 0., 0., -1., 0.
    worst, worst_a, worst_r = 0, 0., 0.
    for start in range(0,count,CHUNK):
        av=a[start:start+CHUNK].to(device='cpu',dtype=torch.float64)
        rv=r[start:start+CHUNK].to(device='cpu',dtype=torch.float64)
        error=(av-rv).abs(); finite=torch.isfinite(av)&torch.isfinite(rv)
        bound=atol+rtol*rv.abs()
        nonfinite+=int((~finite).sum());bad+=int(((error>bound)|~finite).sum())
        if not bool(finite.all()):
            continue
        squared+=float(error.square().sum());reference_squared+=float(rv.square().sum())
        ratio_max=max(ratio_max,float((error/bound).max()))
        value,index=error.max(dim=0)
        if float(value)>maximum:
            maximum=float(value);worst=start+int(index);worst_a=float(av[index]);worst_r=float(rv[index])
    return dict(atol=atol,rtol=rtol,count=count,violations=bad,nonfinite=nonfinite,
        max_abs_error=None if nonfinite else maximum,
        max_normalized_tolerance_ratio=None if nonfinite else ratio_max,
        rmse=None if nonfinite else math.sqrt(squared/count),
        relative_l2=None if nonfinite or not reference_squared else math.sqrt(squared/reference_squared),
        worst_flat_index=worst,reference_at_worst=worst_r,reference_magnitude_at_worst=abs(worst_r),actual_at_worst=worst_a)
