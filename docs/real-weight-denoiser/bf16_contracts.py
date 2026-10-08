"""Pure CPU admission/verdict contracts for a separate, review-gated diagnostic."""
import json
import math
from pathlib import Path

from trace_binding import sha256

PROTOCOL='bf16-application-diagnostic-v1'
PROTOCOL_REVIEW_COMMIT='1e6e0c64f2db5675af1b5ff15d84bef9e011da64'
CAPTURE_MANIFEST='15ef4ca04d95404a0467524015097e306f8dab871eca027635d716b843b469ce'
ULYSSES_SOURCE='8b688762e1a4900e93d1d222828909be5a287c06220ab298ec96c664007b9438'
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
    squared=sum(row.get('squared_error',0) for row in active)
    reference_squared=sum(row.get('reference_squared',0) for row in active)
    count=sum(row['count'] for row in active)
    return dict(rmse=math.sqrt(squared/count) if count else None,
        relative_l2=math.sqrt(squared/reference_squared) if reference_squared else None,count=sum(row['count'] for row in active),violations=sum(row['violations'] for row in active),
        nonfinite=sum(row['nonfinite'] for row in active),
        max_abs_error=max((row['max_abs_error'] or 0) for row in active),
        max_normalized_tolerance_ratio=max((row['max_normalized_tolerance_ratio'] or 0) for row in active))


def require_pass(rows):
    combined=aggregate(rows)
    if combined['violations'] or combined['nonfinite']:raise AssertionError('Unchanged BF16 numerical criterion failed')
    return combined
