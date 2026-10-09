"""Persist identified CPU samples before enforcing the unchanged 80 C threshold."""
import json
import time
from pathlib import Path

THRESHOLD_C=80


def read_cpu_sensors(root=Path('/sys/class/hwmon')):
    rows=[]
    for path in sorted(root.glob('hwmon*/temp*_input')):
        name=(path.parent/'name').read_text().strip()
        if name not in ('k10temp','coretemp'):continue
        label=path.with_name(path.name.replace('_input','_label'))
        row=dict(path=str(path),driver=name,label=label.read_text().strip() if label.exists() else None)
        try:row['celsius']=int(path.read_text())/1000
        except (OSError,ValueError) as exc:row['error']=str(exc)
        rows.append(row)
    return rows


def check_cpu(evidence,phase,sensors=None,limit=THRESHOLD_C,workload=None):
    # Admission may be cooler; runtime callers never raise the fixed threshold.
    if limit>THRESHOLD_C:raise ValueError('Cannot relax CPU threshold')
    if sensors is None:
        try:sensors=read_cpu_sensors()
        except (OSError,ValueError) as exc:sensors=[dict(error=str(exc))]
    valid=bool(sensors) and all('celsius' in row for row in sensors)
    peak=max((row['celsius'] for row in sensors if 'celsius' in row),default=None)
    sample=dict(unix_time=time.time(),phase=phase,workload=workload,sensors=sensors,peak_c=peak,threshold_c=limit,
        accepted=valid and peak<limit)
    with (Path(evidence)/'cpu-thermal.jsonl').open('a') as stream:
        stream.write(json.dumps(sample)+'\n');stream.flush()
    if not sample['accepted']:raise RuntimeError('CPU sensor unavailable or at/above thermal guard; sample persisted')
    return peak
