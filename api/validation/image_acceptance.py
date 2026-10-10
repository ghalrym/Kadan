"""Production HTTP image acceptance and external observation. No inference imports.

Default: one read-only preflight sample. --execute submits one normal request;
use only after hardware clearance and deployment/provenance review. No Redis,
lock, model selection, lifecycle mutation or worker is created by this client.
"""
import argparse
import base64
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request


def stamp():
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), monotonic_ns=time.monotonic_ns())


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} exited {result.returncode}: {result.stderr[-1000:]}')
    return result.stdout


def get_status(url):
    with urllib.request.urlopen(url.rstrip('/') + '/model-lifecycle', timeout=5) as response:
        return json.load(response)


def sample(url, container):
    row = stamp()
    row['errors'] = []
    for name, argv in (
        ('devices', ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free,power.draw,power.limit,temperature.gpu', '--format=csv,noheader,nounits']),
        ('process_memory', ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits']),
        ('process_activity', ['nvidia-smi', 'pmon', '-s', 'u', '-c', '1']),
        ('container_processes', ['docker', 'top', container, '-eo', 'pid,ppid,comm']),
    ):
        try:
            row[name] = command(argv)
        except Exception as error:
            row['errors'].append(f'{name}: {error}')
    row['host_processes'] = []
    pids = set()
    for values in csv.reader(row.get('process_memory', '').splitlines()):
        if len(values) == 3 and values[1].strip().isdigit():
            pids.add(int(values[1]))
    for line in row.get('container_processes', '').splitlines()[1:]:
        fields = line.split()
        if fields and fields[0].isdigit():
            pids.add(int(fields[0]))
    for pid in sorted(pids):
        item = dict(pid=pid)
        try:
            # NSpid maps host PID to the worker PID in container stage logs.
            fields = Path(f'/proc/{pid}/status').read_text().splitlines()
            item['status'] = [line for line in fields if line.split(':')[0] in
                              ('Name', 'PPid', 'NSpid', 'VmRSS', 'VmHWM', 'VmSize')]
            item['start_ticks'] = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        except OSError as error:
            item['unavailable'] = str(error)
        row['host_processes'].append(item)
    try:
        row['admission'] = get_status(url)
    except Exception as error:
        row['errors'].append(f'admission: {error}')
    row['sample_finished'] = stamp()
    return row


def append(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True, help='Operator-reviewed deployment commit; recorded, not assumed verified')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    payload = dict(prompt='A red ball on a white background.', aspect='1:1', count=1, steps=50, seed=0)
    append(args.output / 'events.jsonl', dict(event='preflight', expected_commit=args.expected_commit,
           payload=payload, url=args.url, container=args.container, **stamp()))
    first = sample(args.url, args.container)
    append(args.output / 'telemetry.jsonl', first)
    if not args.execute:
        return
    if first['errors'] or first.get('admission', {}).get('memory') is None:
        raise RuntimeError('Complete telemetry and initialized production admission accounting are required')
    # Save immutable image ID and process identities, without copying secrets/env.
    identity = command(['docker', 'inspect', '--format', '{{.Image}} {{.State.Pid}} {{.State.StartedAt}}', args.container])
    (args.output / 'container-identity.txt').write_text(identity)
    stop = threading.Event()
    def observe():
        while not stop.is_set():
            append(args.output / 'telemetry.jsonl', sample(args.url, args.container))
            stop.wait(1)
    logs = subprocess.Popen(['docker', 'logs', '--timestamps', '--since', first['utc'], '--follow', args.container],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def record_stages():
        for line in logs.stdout:
            if 'worker_stage ' in line:
                append(args.output / 'stages.jsonl', dict(line=line.rstrip(), **stamp()))
    stage_reader = threading.Thread(target=record_stages)
    stage_reader.start()
    observer = threading.Thread(target=observe)
    observer.start()
    try:
        append(args.output / 'events.jsonl', dict(event='http_submit', **stamp()))
        request = urllib.request.Request(args.url.rstrip('/') + '/v1/images/generations',
                   data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=7200) as response:
            result = json.load(response)
        images = result['image'].pop('images_base64')
        if len(images) != 1:
            raise RuntimeError('Expected one image')
        data = base64.b64decode(images[0], validate=True)
        if data[:8] != b'\x89PNG\r\n\x1a\n' or data[16:24] != (2048).to_bytes(4, 'big') * 2:
            raise RuntimeError('Expected a 2048x2048 PNG')
        (args.output / 'output.png').write_bytes(data)
        append(args.output / 'events.jsonl', dict(event='http_completed', response=result,
               sha256=hashlib.sha256(data).hexdigest(), visual_quality='not_yet_reviewed', **stamp()))
    except BaseException as error:
        append(args.output / 'events.jsonl', dict(event='request_failed', error=repr(error), **stamp()))
        raise
    finally:
        stop.set()
        observer.join(timeout=50)
        logs.terminate()  # Only this read-only log follower, never the API/container.
        logs.wait(timeout=10)
        stage_reader.join(timeout=5)
        append(args.output / 'telemetry.jsonl', sample(args.url, args.container))
        append(args.output / 'events.jsonl', dict(event='observation_ended', **stamp()))


if __name__ == '__main__':
    main()
