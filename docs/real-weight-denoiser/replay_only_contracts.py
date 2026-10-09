"""Pin the retained incomplete attempt; no capture, mutation or context deletion."""
import json
from pathlib import Path

from holdout_contracts import verify_contexts, safe_file, require_prior_step20
from trace_binding import sha256

PROTOCOL='bf16-step39-replay-only-v1'
CAPTURE_COMMIT='f0b4eefc615a6370fb6bb43a13264a0128e0ea13'
CONTEXT_SHA='9ceff1294a8aecf4a3211ab952d5b8f3beeec151fc8640431d536e2257c0fafa'


def require_review(record,commit):
    if record.get('protocol')!=PROTOCOL or record.get('source_commit')!=commit or record.get('context_manifest_sha256')!=CONTEXT_SHA:
        raise ValueError('Review must bind exact replay source and retained context')
    if record.get('decision')!='approved-for-bounded-execution' or not record.get('reviewer') or not record.get('review_reference'):
        raise ValueError('Independent review and execution authorization required')


def verify_retained(root,base,step20):
    root=Path(root);require_prior_step20(step20)
    contexts=root/'contexts'
    if sha256(contexts/'manifest.json')!=CONTEXT_SHA:raise ValueError('Wrong retained context manifest')
    manifest=verify_contexts(contexts,base,39,CAPTURE_COMMIT)
    owner=json.loads(safe_file(contexts,'ownership.json').read_text())
    names=[row['file'] for row in [*manifest['blocks'],manifest['tail']]]
    if owner['run_id']!=manifest['run_id'] or set(owner['files'])!=set(names) or set(p.name for p in contexts.iterdir())!=set(names+['manifest.json','ownership.json']):
        raise ValueError('Retained context ownership mismatch')
    if json.loads((root/'pause-result.json').read_text()).get('restored') is not True:
        raise ValueError('Prior attempt restoration unconfirmed')
    for rank in (0,1):
        stage=root/'holdout-replay-evidence'
        status=json.loads((stage/f'verdict-rank-{rank}.json').read_text())
        identity=json.loads((stage/f'identity-rank-{rank}.json').read_text())
        if status.get('step_index')!=39 or status.get('heldout_step')!='incomplete' or identity.get('context_manifest_sha256')!=CONTEXT_SHA:
            raise ValueError('Expected exact retained incomplete attempt')
    return manifest
