"""Persist every CPU reading before enforcing the reviewed-host test policy."""
import json
import logging
import math
import time
from pathlib import Path

from thermal_policy import ABORT_C, WARNING_C, evaluate, read_identity

THRESHOLD_C=ABORT_C


from thermal_policy import read_cpu_sensors, CpuMonitor

_monitor=CpuMonitor()


def check_cpu(evidence,phase,sensors=None,limit=THRESHOLD_C,workload=None,*,identity=None):
    if sensors is None and identity is None:
        # Both entry points share inventory, mapping and freshness enforcement.
        sample=_monitor.sample(limit)
    else:
        if sensors is None:
            try:sensors=read_cpu_sensors()
            except (OSError,ValueError) as exc:sensors=[dict(error=str(exc))]
        if identity is None:
            try:identity=read_identity()
            except (OSError,ValueError) as exc:identity=dict(error=str(exc))
        assessment=evaluate(sensors,identity,limit)
        peak=max((row['celsius'] for row in sensors if isinstance(row.get('celsius'),(int,float))
                  and math.isfinite(row['celsius'])),default=None)
        sample=dict(sensors=sensors,identity=identity,peak_c=peak,**assessment)
    sample.update(unix_time=time.time(),phase=phase,workload=workload,threshold_c=sample['abort_c'])
    with (Path(evidence)/'cpu-thermal.jsonl').open('a') as stream:
        stream.write(json.dumps(sample)+'\n');stream.flush()
    if not sample['accepted']:
        raise RuntimeError('CPU sensor mapping unavailable, invalid or at/above thermal guard; sample persisted')
    if sample['warning']:
        logging.getLogger(__name__).warning('CPU software test warning: peak=%s C warning=%s abort=%s policy=%s',
                                            sample['peak_c'],sample['warning_c'],sample['abort_c'],sample['policy'])
    return sample['peak_c']
