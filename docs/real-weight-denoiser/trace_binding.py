"""Verify capture/trace/backend identities before comparing saved tensors."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):result.update(chunk)
    return result.hexdigest()


def verify_binding(capture,traces):
    capture=Path(capture);traces=Path(traces)
    binding=json.loads((traces/'trace-binding.json').read_text())
    if binding['protocol']!='block7-traces-v1':raise ValueError('Unknown trace protocol')
    if sha256(capture/'manifest.json')!=binding['capture_manifest_sha256']:
        raise ValueError('Capture manifest identity mismatch')
    manifest=json.loads((capture/'manifest.json').read_text())
    row=manifest['blocks'][7]
    if row['sha256']!=binding['block_sha256'] or sha256(capture/row['file'])!=row['sha256']:
        raise ValueError('Captured block identity mismatch')
    if binding['producer_source_commit']!='1cc8c05e84d17cfa7e9f730510db01842063b546':
        raise ValueError('Unexpected diagnostic source identity')
    for backend in ('default','math'):
        entry=binding['backends'][backend]
        if entry['backend']!=backend:raise ValueError('Backend identity mismatch')
        for name in (f'{backend}-intermediates.pt',f'{backend}-report.json','settings.json'):
            if sha256(traces/name)!=entry['files'][name]:raise ValueError('Trace hash mismatch: '+name)
        report=json.loads((traces/f'{backend}-report.json').read_text())
        if report['backend']!=backend or report['operators']!=entry['operators']:
            raise ValueError('Backend report mismatch')
    return binding
