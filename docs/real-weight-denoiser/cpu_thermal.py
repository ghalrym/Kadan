"""Reviewed-host software test policy; no hardware/firmware writes.

Only this verified CPU/module/sensor mapping receives the 85/90 C policy.
Other CPUs retain an explicit 75/80 C conservative software policy. Unknown
sensor roles, missing inputs and stale acquisitions always fail closed.
"""
import math
from pathlib import Path
import platform
import re
import time

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
                kernel=platform.release(),driver_srcversion=module.read_text().strip() if module.exists() else None)


def read_cpu_sensors(root=Path('/sys/class/hwmon')):
    rows=[]
    for path in sorted(root.glob('hwmon*/temp*_input')):
        try:name=(path.parent/'name').read_text().strip()
        except OSError as exc:
            row=dict(path=str(path),driver=None,label=None,error=str(exc))
            try:row['celsius']=int(path.read_text())/1000
            except (OSError,ValueError) as value_error:row['error']+='; '+str(value_error)
            rows.append(row);continue
        if name not in ('k10temp','coretemp'):continue
        label=path.with_name(path.name.replace('_input','_label'))
        row=dict(path=str(path),driver=name)
        device=path.parent/'device'
        row['device']=device.resolve().name if device.exists() else path.parent.resolve().name
        fields=[('celsius',path),('label',label)]
        if name=='k10temp':fields.extend([('pci_vendor',device/'vendor'),('pci_device',device/'device')])
        for key,source in fields:
            try:
                value=source.read_text().strip()
                row[key]=int(value)/1000 if key=='celsius' else value
            except (OSError,ValueError) as exc:
                row['error']=row.get('error','')+str(exc)+'; '
        rows.append(row)
    return rows



def evaluate(sensors,identity,limit=ABORT_C):
    if not isinstance(limit,(int,float)) or not math.isfinite(limit) or not 0<limit<=ABORT_C:
        raise ValueError('Invalid or relaxed CPU threshold')
    verified=identity==IDENTITY
    warning=WARNING_C if verified else 75
    abort=min(limit,ABORT_C if verified else 80)
    errors=[];results=[];labels=[];main=False
    if not identity or 'error' in identity or identity.get('vendor') not in ('AuthenticAMD','GenuineIntel'):errors.append('unavailable_cpu_identity')
    for sensor in sensors:
        label=sensor.get('label');path=Path(sensor.get('path') or '')
        ccd=re.fullmatch(r'Tccd([1-8])',label or '')
        expected='temp1_input' if label=='Tctl' else 'temp2_input' if label=='Tdie' else ('temp'+str(int(ccd[1])+2)+'_input' if ccd else None)
        driver=sensor.get('driver')
        amd=(driver=='k10temp' and identity.get('vendor')=='AuthenticAMD' and expected is not None and path.name==expected)
        intel=(driver=='coretemp' and identity.get('vendor')=='GenuineIntel' and bool(re.fullmatch(r'(Core|Package id) [0-9]+',label or ''))
               and bool(re.fullmatch(r'temp[1-9][0-9]*_input',path.name)))
        mapped=amd or intel
        if verified:
            mapped=(amd and label!='Tdie' and sensor.get('pci_vendor')=='0x1022' and sensor.get('pci_device')=='0x1653'
                    and sensor.get('device')=='0000:00:18.3')
        main=main or (mapped and (label=='Tctl' or driver=='coretemp'))
        value=sensor.get('celsius')
        valid=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=150
        if not mapped:errors.append('unmapped_sensor:'+str(label))
        if not valid or 'error' in sensor:errors.append('unreadable_or_invalid_sensor:'+str(label))
        labels.append((sensor.get('device'),driver,label))
        results.append(dict(label=label,path=str(path),kind='control' if label=='Tctl' else 'ccd' if ccd else 'die' if label=='Tdie' else 'core' if intel else 'unknown',
            mapped=mapped,warning_c=warning,abort_c=abort,
            state='invalid' if not valid or not mapped or 'error' in sensor else 'abort' if value>=abort else 'warning' if value>=warning else 'normal'))
    if verified and not REQUIRED.issubset(sensor.get('label') for sensor in sensors):errors.append('missing_required_sensor')
    if not main:errors.append('missing_main_sensor')
    if len(labels)!=len(set(labels)):errors.append('duplicate_sensor_identity')
    return dict(policy=POLICY_ID if verified else 'unverified-cpu-conservative-75-80-v1',
                mapping_verified=verified,mapping_note=None if verified else 'No reviewed 5955WX mapping; conservative software limit, not an assumed junction margin',
                warning_c=warning,abort_c=abort,assessments=results,mapping_errors=errors,
                warning=any(r['state']=='warning' for r in results),
                accepted=not errors and bool(results) and all(r['state'] not in ('abort','invalid') for r in results))


class CpuMonitor:
    """Fresh reads with fixed per-owner inventory; no threads or GPU dependencies.

    Acquisition age is checked before returning a reading. The external test
    watchdog independently interrupts a blocked read at its existing deadline.
    """
    freshness=1.0

    def __init__(self):
        self.inventory=None

    def sample(self,limit=ABORT_C):
        started=time.monotonic()
        try:sensors=read_cpu_sensors()
        except (OSError,ValueError) as exc:sensors=[dict(error=str(exc))]
        try:identity=read_identity()
        except (OSError,ValueError) as exc:identity=dict(error=str(exc))
        result=evaluate(sensors,identity,limit)
        inventory=frozenset((s.get('device'),s.get('driver'),s.get('label'),Path(s.get('path') or '').name) for s in sensors)
        if self.inventory is not None and inventory!=self.inventory:
            result['mapping_errors'].append('sensor_inventory_changed');result['accepted']=False
        finished=time.monotonic()
        if finished-started>=self.freshness:
            result['mapping_errors'].append('stale_cpu_acquisition');result['accepted']=False
        if result['accepted'] and self.inventory is None:self.inventory=inventory
        peak=max((s['celsius'] for s in sensors if isinstance(s.get('celsius'),(int,float)) and math.isfinite(s['celsius'])),default=None)
        return dict(sensors=sensors,identity=identity,peak_c=peak,acquisition_started=started,
                    acquisition_finished=finished,acquisition_seconds=finished-started,**result)
