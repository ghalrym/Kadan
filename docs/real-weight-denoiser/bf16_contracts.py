"""Pure CPU admission/verdict contracts for a separate, review-gated diagnostic."""
import json
import math
from pathlib import Path

from trace_binding import sha256

PROTOCOL='bf16-application-diagnostic-v1'
PROTOCOL_REVIEW_COMMIT='1e6e0c64f2db5675af1b5ff15d84bef9e011da64'
CAPTURE_MANIFEST='15ef4ca04d95404a0467524015097e306f8dab871eca027635d716b843b469ce'
ULYSSES_SOURCE='d80db3f89f237552f7759eae57d032d06f001c222beda9b8883731bc9dc1d297'
ATOL=RTOL=.02
LIMIT=32*1024**3


def verify_capture(root):
    root=Path(root)
    if sha256(root/'manifest.json')!=CAPTURE_MANIFEST:raise ValueError('Expected existing reviewed capture')
    manifest=json.loads((root/'manifest.json').read_text())
    if len(manifest['blocks'])!=32:raise ValueError('Incomplete capture')
    for row in [*manifest['blocks'],manifest['tail']]:
        path=root/row['file']
        if path.parent.resolve()!=root.resolve() or sha256(path)!=row['sha256']:
            raise ValueError('Capture packet identity mismatch')
    return manifest


def require_review(record,commit):
    if record.get('protocol')!=PROTOCOL or record.get('source_commit')!=commit:
        raise ValueError('Review must name this protocol and exact source commit')
    if record.get('decision')!='approved-for-bounded-execution' or not record.get('reviewer') or not record.get('review_reference'):
        raise ValueError('Independent implementation review and execution authorization required')


def require_ci(runs,commit):
    for name in ('API tests','Native worker CPU tests'):
        matches=[run for run in runs if run['name']==name and run['headSha']==commit and run['event']=='push']
        if not matches or any(run['status']!='completed' or run['conclusion']!='success' for run in matches):
            raise ValueError('Exact-head green CI required: '+name)


def verdict(first_slice,timing='not-run'):
    return dict(protocol=PROTOCOL,first_slice=first_slice,overall_protocol='incomplete',
        later_steps={'20':'not-run','39':'not-run'},fp32_protocol='failed',
        rejected_candidate_v1='rejected',timing=timing)


def aggregate(rows):
    active=[row for row in rows if row is not None]
    if not active:raise ValueError('Missing rank evidence')
    count=sum(row['count'] for row in active)
    nonfinite=sum(row['nonfinite'] for row in active)
    edges=active[0]['error_histogram_edges']
    if any(row['error_histogram_edges']!=edges for row in active):
        raise ValueError('Rank histogram edges disagree')
    counts=[sum(row['error_histogram_counts'][index] for row in active) for index in range(len(edges)-1)]
    finite_count=sum(row['finite_count'] for row in active)
    if sum(counts)!=finite_count:raise ValueError('Histogram finite count mismatch')
    quantiles={}
    if finite_count:
        for q in (.5,.9,.99,1.):
            target=max(1,q*finite_count);cumulative=0
            for index,value in enumerate(counts):
                cumulative+=value
                if cumulative>=target:
                    quantiles[str(q)]=[edges[index],edges[index+1]]
                    break
    squared=sum(row['squared_error'] for row in active)
    reference_squared=sum(row['reference_squared'] for row in active)
    points=[dict(row['max_normalized_point'],rank=row['rank']) if 'rank' in row else dict(row['max_normalized_point'])
        for row in active if row['max_normalized_point'] is not None]
    return dict(count=count,violations=sum(row['violations'] for row in active),nonfinite=nonfinite,
        finite_count=finite_count,
        rmse=None if nonfinite or not count else math.sqrt(squared/count),
        relative_l2=None if nonfinite or not reference_squared else math.sqrt(squared/reference_squared),
        max_abs_error=None if nonfinite else max(row['max_abs_error'] for row in active),
        max_normalized_tolerance_ratio=None if nonfinite else max(row['max_normalized_tolerance_ratio'] for row in active),
        max_normalized_point=None if nonfinite or not points else max(points,key=lambda point:point['ratio']),
        absolute_error_quantile_bins=None if nonfinite else quantiles,
        finite_error_histogram=dict(edges=edges,counts=counts,scope='finite-elements-only'),
        error_metrics_status='unavailable-nonfinite-input' if nonfinite else 'complete')


def require_pass(rows):
    combined=aggregate(rows)
    if combined['violations'] or combined['nonfinite']:raise AssertionError('Unchanged BF16 numerical criterion failed')
    return combined


def timing_admission(remaining_by_rank,minimum_seconds=120):
    """All ranks branch on the same conservative, control-group gathered budget."""
    if len(remaining_by_rank)!=2 or any(not math.isfinite(value) for value in remaining_by_rank):
        raise ValueError('Two finite rank budgets required')
    remaining=min(remaining_by_rank)
    return dict(admitted=remaining>=minimum_seconds,minimum_remaining_seconds=remaining,
        required_seconds=minimum_seconds,remaining_by_rank=list(remaining_by_rank))
