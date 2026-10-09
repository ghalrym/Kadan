"""Frozen single-case trajectory criteria, order and storage envelope."""
import json
from pathlib import Path

from bf16_contracts import LIMIT
REVISION='d26bb61231c349cf6b7896fa83353113880e1ba3'
PROMPT='A single red apple on a plain white table, soft natural daylight, realistic still-life photograph.'
from trace_binding import sha256

PROTOCOL='full-independent-trajectory-v2-rgba'
CASES=('reference','repeat','candidate')
SETTINGS=dict(checkpoint=REVISION,seed=42,prompt=PROMPT,width=2048,height=2048,steps=40,
    dtype='bfloat16',true_cfg_scale=1.0,use_kv_cache=True,compilation=False,component_offload=True)
CRITERIA=dict(latent_atol=.02,latent_rtol=.02,float_atol=2/255,float_mae=.5/255,pixel_max=2,pixel_mae=.5,violations=0,nonfinite=0)
# 81 BF16 tensors: initial + prediction/post-latent at each step, each 16384x64.
# RGBA float32 + uint8 + conservative lossless PNG cap; serialization cap per tensor.
PER_RUN_PLAN=dict(step_tensor_bytes=81*16384*64*2,float_rgba_bytes=2048*2048*4*4,
    uint8_rgba_bytes=2048*2048*4,png_cap_bytes=24*1024**2,serialization_margin_bytes=83*65536)
RUN_CAP=sum(PER_RUN_PLAN.values())
FAILURE_RESERVE=256*1024**2
LOG_RESERVE=128*1024**2


def storage_plan(existing):
    total=existing+3*RUN_CAP+FAILURE_RESERVE+LOG_RESERVE
    if total>LIMIT:raise ValueError('Full trajectory artifacts do not fit fixed shared budget')
    return dict(existing_bytes=existing,per_run=PER_RUN_PLAN,per_run_cap=RUN_CAP,runs=3,
        failure_reserve=FAILURE_RESERVE,log_reserve=LOG_RESERVE,projected_total=total,limit=LIMIT)


class StepOrder:
    def __init__(self):self.next=0;self.pending=False
    def prediction(self,index,mode):
        if index!=self.next or self.pending or mode!=('extract' if index==0 else 'cached') or index>=40:
            raise ValueError('Unexpected transformer/scheduler order')
        self.pending=True
    def scheduler(self):
        if not self.pending:raise ValueError('Scheduler has no independent prediction')
        index=self.next;self.next+=1;self.pending=False;return index
    def complete(self):
        if self.next!=40 or self.pending:raise ValueError('Incomplete trajectory')


def require_review(record,commit,case):
    if case not in CASES or record.get('protocol')!=PROTOCOL or record.get('source_commit')!=commit or record.get('case')!=case:
        raise ValueError('Review must bind exact source and case')
    if record.get('criteria')!=CRITERIA or record.get('settings')!=SETTINGS:
        raise ValueError('Independent protocol review must freeze settings and criteria')
    if record.get('decision')!='approved-for-bounded-execution' or not record.get('reviewer') or not record.get('review_reference'):
        raise ValueError('Independent protocol/source review and execution authorization required')


def safe(root,name):
    path=Path(root)/name
    if path.is_symlink() or Path(name).name!=name or path.resolve().parent!=Path(root).resolve():raise ValueError('Unsafe artifact path')
    return path


def verify_run(root,case,commit):
    root=Path(root);manifest=json.loads(safe(root,'manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL or manifest['case']!=case or manifest['source_commit']!=commit or manifest['settings']!=SETTINGS or manifest['criteria']!=CRITERIA:
        raise ValueError('Prior run identity mismatch')
    if manifest.get('raw_vae_finite') is not True or manifest['status']!='passed' or manifest['completed_steps']!=40:raise ValueError('Prior full run did not pass')
    expected={'initial.pt','float-rgba.pt','pixels.pt','output.png','input-identity.json'}
    expected|={f'{kind}-{step:02}.pt' for kind in ('prediction','latent') for step in range(40)}
    names=[row['file'] for row in manifest['artifacts']]
    if len(names)!=len(set(names)) or set(names)!=expected:raise ValueError('Incomplete/duplicate full trajectory artifacts')
    for row in manifest['artifacts']:
        path=safe(root,row['file'])
        if sha256(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Changed trajectory artifact')
    if case!='reference':
        reports=manifest.get('comparison_reports',[])
        expected=[f'comparisons-rank-{rank}.jsonl' for rank in range(2 if case=='candidate' else 1)]
        if [row['file'] for row in reports]!=expected or manifest.get('comparisons_count')!=83:raise ValueError('Missing complete comparison reports')
        for row in reports:
            path=safe(root,row['file'])
            if sha256(path)!=row['sha256']:raise ValueError('Comparison report changed')
            values=[json.loads(line) for line in path.read_text().splitlines()]
            if len(values)!=83 or not all(value['passed'] for value in values):raise ValueError('Prior full comparisons did not all pass')
    return manifest


def require_predecessors(case,commit,reference=None,repeat=None):
    if case=='reference':return
    if reference is None:raise ValueError('Eager reference is required')
    reference=Path(reference);verify_run(reference/'trajectory-evidence','reference',commit)
    if json.loads((reference/'pause-result.json').read_text()).get('restored') is not True:raise ValueError('Reference restoration incomplete')
    if case=='candidate':
        if repeat is None:raise ValueError('Eager repeatability must pass before candidate admission')
        repeat=Path(repeat);manifest=verify_run(repeat/'trajectory-evidence','repeat',commit)
        if json.loads((repeat/'pause-result.json').read_text()).get('restored') is not True:raise ValueError('Repeat restoration incomplete')
        if manifest.get('reference_manifest_sha256')!=sha256(reference/'trajectory-evidence/manifest.json'):
            raise ValueError('Eager repeat did not validate this reference')
