"""Reviewed-host software test policy; no hardware/firmware writes.

Only this verified CPU/module/sensor mapping receives the 85/90 C policy.
Unknown mapping is an error, never a reason to discard a sensor or assume Tdie.
"""
import math
from pathlib import Path
import platform
import re

POLICY_ID='5955wx-k10temp-85-90-v1'
WARNING_C=85
ABORT_C=90
IDENTITY=dict(vendor='AuthenticAMD',model_name='AMD Ryzen Threadripper PRO 5955WX 16-Cores',
              family='25',model='8',kernel='6.17.9-76061709-generic',
              driver_srcversion='A594DFB9AAE185D37A6B0D8')
REQUIRED={'Tctl','Tccd3','Tccd5'}


def read_identity(cpuinfo=Path('/proc/cpuinfo'),module=Path('/sys/module/k10temp/srcversion')):
    chips=[]
    for section in cpuinfo.read_text().strip().split('\n\n'):
        fields=dict(line.split(':',1) for line in section.splitlines() if ':' in line)
        fields={k.strip():v.strip() for k,v in fields.items()}
        chips.append(tuple(fields.get(k) for k in ('vendor_id','model name','cpu family','model')))
    if not chips or len(set(chips))!=1:raise ValueError('CPU identity missing or heterogeneous')
    return dict(zip(('vendor','model_name','family','model'),chips[0]),
                kernel=platform.release(),driver_srcversion=module.read_text().strip())


def evaluate(sensors,identity,limit=ABORT_C):
    errors=[];results=[];labels=[]
    if identity!=IDENTITY:errors.append('unreviewed_cpu_or_kernel_mapping')
    for sensor in sensors:
        label=sensor.get('label');path=Path(sensor.get('path',''))
        ccd=re.fullmatch(r'Tccd([1-8])',label or '')
        expected='temp1_input' if label=='Tctl' else ('temp'+str(int(ccd[1])+2)+'_input' if ccd else None)
        mapped=(sensor.get('driver')=='k10temp' and expected is not None and path.name==expected
                and sensor.get('pci_vendor')=='0x1022' and sensor.get('pci_device')=='0x1653'
                and sensor.get('device')=='0000:00:18.3')
        value=sensor.get('celsius')
        valid=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=150
        if not mapped:errors.append('unmapped_sensor:'+str(label))
        if not valid or 'error' in sensor:errors.append('unreadable_or_invalid_sensor:'+str(label))
        labels.append(label)
        results.append(dict(label=label,path=str(path),kind='control' if label=='Tctl' else 'ccd' if ccd else 'unknown',
            mapped=mapped,warning_c=WARNING_C,abort_c=limit,
            state='invalid' if not valid or not mapped or 'error' in sensor else 'abort' if value>=limit else 'warning' if value>=WARNING_C else 'normal'))
    if not REQUIRED.issubset(labels):errors.append('missing_required_sensor')
    if len(labels)!=len(set(labels)):errors.append('duplicate_sensor_identity')
    return dict(policy=POLICY_ID,warning_c=WARNING_C,abort_c=limit,assessments=results,mapping_errors=errors,
                warning=any(r['state']=='warning' for r in results),
                accepted=not errors and bool(results) and all(r['state'] not in ('abort','invalid') for r in results))
