"""Held-out step admission, immutable packet references and owned artifact cleanup."""
import json
import math
import os
from pathlib import Path

from bf16_contracts import CAPTURE_MANIFEST, LIMIT
from trace_binding import sha256

PROTOCOL='bf16-heldout-steps-v1'
STEPS=(20,39)
BLOCK_KEYS={'hidden','expected','modulation','rotary','prefix','key_valid','step_index','base_packet_sha256'}
TAIL_KEYS={'temb','target_mask','expected','step_index','base_packet_sha256'}
RESERVE=512*1024**2


def require_step(step):
    if step not in STEPS:raise ValueError('Only predeclared zero-based steps 20 and 39 are supported')


def require_review(record,commit,step):
    require_step(step)
    if record.get('protocol')!=PROTOCOL or record.get('source_commit')!=commit or record.get('step_index')!=step:
        raise ValueError('Review must identify exact held-out step and source commit')
    if record.get('decision')!='approved-for-bounded-execution' or not record.get('reviewer') or not record.get('review_reference'):
        raise ValueError('Independent implementation review and execution approval required')


def safe_file(root,name):
    root=Path(root);path=root/name
    if Path(name).name!=name or path.is_symlink() or path.resolve().parent!=root.resolve():
        raise ValueError('Artifact must be an owned direct child')
    return path


def merge_context(base,context,expected_sha,step,tail=False):
    require_step(step)
    if set(context)!=(TAIL_KEYS if tail else BLOCK_KEYS):raise ValueError('Context schema forbids weight packets or unexpected fields')
    if context['step_index']!=step or context['base_packet_sha256']!=expected_sha:
        raise ValueError('Context/weight/step identity mismatch')
    # The immutable mmap-backed weight tensors are referenced, never serialized again.
    return dict(base,**{key:value for key,value in context.items() if key not in ('step_index','base_packet_sha256')})


def verify_contexts(root,base_manifest,step,source_commit):
    require_step(step);root=Path(root)
    manifest=json.loads(safe_file(root,'manifest.json').read_text())
    if manifest['protocol']!=PROTOCOL or manifest['step_index']!=step or manifest['source_commit']!=source_commit or manifest['base_manifest_sha256']!=CAPTURE_MANIFEST:
        raise ValueError('Held-out capture provenance mismatch')
    for key in ('seed','prompt','size','steps','dtype','scheduler_config'):
        if manifest[key]!=base_manifest[key]:raise ValueError('Production setting changed: '+key)
    times=manifest['observed_timesteps']
    if len(times)!=step+1 or any(len(value)!=1 or not math.isfinite(value[0]) for value in times):
        raise ValueError('Incomplete timestep provenance')
    if [row['index'] for row in manifest['blocks']]!=list(range(32)):
        raise ValueError('Incomplete/duplicate block capture')
    for index,row in enumerate([*manifest['blocks'],manifest['tail']]):
        base=base_manifest['blocks'][index] if index<32 else base_manifest['tail']
        expected=f'block-{index:02}.context.pt' if index<32 else 'tail.context.pt'
        if row['file']!=expected or row['base_packet_sha256']!=base['sha256'] or sha256(safe_file(root,row['file']))!=row['sha256']:
            raise ValueError('Context packet identity mismatch')
    return manifest


def verdict(step,state,context_sha):
    return dict(protocol=PROTOCOL,step_index=step,heldout_step=state,context_manifest_sha256=context_sha,
        first_slice='passed-67b3768',fp32_protocol='failed',rejected_candidate_v1='rejected',
        full_trajectory='not-run',decoded_image='not-run',production_activation=False)


def require_prior_step20(root):
    root=Path(root)
    retained=root/'retained-context-manifest.json'
    digest=sha256(retained)
    manifest=json.loads(retained.read_text())
    receipt=json.loads((root/'context-cleanup.json').read_text())
    restored=json.loads((root/'pause-result.json').read_text())
    names=[row['file'] for row in [*manifest['blocks'],manifest['tail']]]
    if manifest.get('step_index')!=20 or receipt.get('step_index')!=20 or receipt.get('context_manifest_sha256')!=digest or receipt.get('deleted_owned_files')!=names:
        raise ValueError('Step 20 cleanup must bind the retained context manifest')
    if restored.get('restored') is not True or list((root/'contexts').iterdir()):
        raise ValueError('Step 20 restoration and context release must be complete')
    for rank in (0,1):
        row=json.loads((root/'holdout-replay-evidence'/f'verdict-rank-{rank}.json').read_text())
        if row.get('protocol')!=PROTOCOL or row.get('step_index')!=20 or row.get('heldout_step')!='passed' or row.get('context_manifest_sha256')!=digest:
            raise ValueError('Step 39 requires successful reviewed step 20')


def cleanup_successful_contexts(root,evidence):
    """Delete only this run's exact successful contexts, after durable manifest copy."""
    root=Path(root);evidence=Path(evidence)
    manifest_path=safe_file(root,'manifest.json');digest=sha256(manifest_path)
    manifest=json.loads(manifest_path.read_text());owner=json.loads(safe_file(root,'ownership.json').read_text())
    if owner['run_id']!=manifest['run_id']:raise ValueError('Ownership mismatch')
    rows=[*manifest['blocks'],manifest['tail']]
    names=[row['file'] for row in rows]
    if set(owner['files'])!=set(names) or set(p.name for p in root.iterdir())!=set(names+['manifest.json','ownership.json']):
        raise ValueError('Unowned or incomplete artifacts present')
    for rank in (0,1):
        status=json.loads((evidence/'holdout-replay-evidence'/f'verdict-rank-{rank}.json').read_text())
        if status.get('heldout_step')!='passed' or status.get('context_manifest_sha256')!=digest or status.get('step_index')!=manifest['step_index']:
            raise ValueError('Both exact-context replay verdicts must pass before cleanup')
    for row in rows:
        if sha256(safe_file(root,row['file']))!=row['sha256']:raise ValueError('Refuse changed context cleanup')
    with (evidence/'retained-context-manifest.json').open('x') as stream:
        stream.write(manifest_path.read_text());stream.flush();os.fsync(stream.fileno())
    for name in names:safe_file(root,name).unlink()
    manifest_path.unlink();safe_file(root,'ownership.json').unlink()
    receipt=dict(context_manifest_sha256=digest,step_index=manifest['step_index'],deleted_owned_files=names)
    (evidence/'context-cleanup.json').write_text(json.dumps(receipt,indent=2))
