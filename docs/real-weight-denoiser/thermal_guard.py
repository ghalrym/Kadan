"""Persist every CPU reading before enforcing the reviewed-host test policy."""
import json
import logging
import math
import time
from pathlib import Path

from thermal_policy import ABORT_C, WARNING_C, evaluate, read_identity

THRESHOLD_C=ABORT_C


def read_cpu_sensors(root=Path('/sys/class/hwmon')):
    rows=[]
    for path in sorted(root.glob('hwmon*/temp*_input')):
        try:name=(path.parent/'name').read_text().strip()
        except OSError as exc:
            rows.append(dict(path=str(path),error=str(exc)));continue
        if name not in ('k10temp','coretemp'):continue
        label=path.with_name(path.name.replace('_input','_label'))
        row=dict(path=str(path),driver=name)
        device=path.parent/'device'
        row['device']=device.resolve().name
        for key,source in [('celsius',path),('label',label),('pci_vendor',device/'vendor'),('pci_device',device/'device')]:
            try:
                value=source.read_text().strip()
                row[key]=int(value)/1000 if key=='celsius' else value
            except (OSError,ValueError) as exc:
                row['error']=row.get('error','')+str(exc)+'; '
        rows.append(row)
    return rows


def check_cpu(evidence,phase,sensors=None,limit=THRESHOLD_C,workload=None,*,identity=None):
    # Cooler admission is allowed. Unknown hardware never inherits a raised limit.
    if not isinstance(limit,(int,float)) or not math.isfinite(limit) or not 0<limit<=THRESHOLD_C:
        raise ValueError('Invalid or relaxed CPU threshold')
    if sensors is None:
        try:sensors=read_cpu_sensors()
        except (OSError,ValueError) as exc:sensors=[dict(error=str(exc))]
    if identity is None:
        try:identity=read_identity()
        except (OSError,ValueError) as exc:identity=dict(error=str(exc))
    assessment=evaluate(sensors,identity,limit)
    peak=max((row['celsius'] for row in sensors if isinstance(row.get('celsius'),(int,float))
              and math.isfinite(row['celsius'])),default=None)
    sample=dict(unix_time=time.time(),phase=phase,workload=workload,sensors=sensors,identity=identity,
                peak_c=peak,threshold_c=limit,**assessment)
    with (Path(evidence)/'cpu-thermal.jsonl').open('a') as stream:
        stream.write(json.dumps(sample)+'\n');stream.flush()
    if not sample['accepted']:
        raise RuntimeError('CPU sensor mapping unavailable, invalid or at/above thermal guard; sample persisted')
    if sample['warning']:
        logging.getLogger(__name__).warning('CPU software test warning: peak=%s C warning=%s abort=%s policy=%s',
                                            peak,WARNING_C,limit,sample['policy'])
    return peak
