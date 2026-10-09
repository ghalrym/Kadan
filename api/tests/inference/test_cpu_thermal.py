"""Same host/transport cutoffs, without GPU access or child model startup."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.inference import cpu_thermal as cpu
from api.inference.image.rank_transport import ProcessRanks

# Host validation scripts live outside the API package. Import that entry point
# to test the actual shared policy, not a duplicate formula in this test.
sys.path.insert(0,str(Path(__file__).resolve().parent/'../../..'/'docs/real-weight-denoiser'))
import thermal_guard


def sensors(value=50):
    return [dict(driver='k10temp',label=label,path='/sys/class/hwmon/hwmon3/'+name,
                 celsius=value,pci_vendor='0x1022',pci_device='0x1653',device='0000:00:18.3')
            for label,name in [('Tctl','temp1_input'),('Tccd3','temp5_input'),('Tccd5','temp7_input')]]


class CpuThermalTests(unittest.TestCase):
    def transport(self):
        return ProcessRanks('/unused',SimpleNamespace(execution_bytes=1,context_bytes=1,host_bytes=1),
                            guard=lambda:None,memory_probe=lambda:{})

    def test_host_and_transport_share_every_boundary(self):
        for index in range(3):
            for value,accepted,warning in [(79.999,True,False),(80,True,False),(80.75,True,False),
                    (84.999,True,False),(85,True,True),(89.999,True,True),(90,False,False),(90.001,False,False)]:
                with self.subTest(sensor=index,value=value),tempfile.TemporaryDirectory() as root:
                    data=sensors();data[index]['celsius']=value;transport=self.transport()
                    with patch.object(cpu,'read_cpu_sensors',return_value=data),patch.object(cpu,'read_identity',return_value=cpu.IDENTITY), \
                         patch.object(thermal_guard,'_monitor',cpu.CpuMonitor()), \
                         patch('api.inference.image.rank_transport.subprocess.run',return_value=SimpleNamespace(stdout='50\n')) as gpu:
                        if accepted:
                            self.assertEqual(thermal_guard.check_cpu(root,'test'),value);transport._guard();gpu.assert_called_once()
                        else:
                            with self.assertRaises(RuntimeError):thermal_guard.check_cpu(root,'test')
                            with self.assertRaises(RuntimeError):transport._guard()
                            gpu.assert_not_called()
                    record=json.loads((Path(root)/'cpu-thermal.jsonl').read_text())
                    self.assertEqual(record['warning'],warning);self.assertEqual(record['abort_c'],90)
                    self.assertEqual(record['sensors'],data)

    def test_unknown_hardware_has_explicit_conservative_80_cutoff(self):
        identity=dict(cpu.IDENTITY,model_name='Unreviewed AMD CPU')
        for value,accepted in [(74.999,True),(75,True),(79.999,True),(80,False)]:
            with tempfile.TemporaryDirectory() as root,patch.object(cpu,'read_identity',return_value=identity), \
                 patch.object(cpu,'read_cpu_sensors',return_value=sensors(value)), \
                 patch.object(thermal_guard,'_monitor',cpu.CpuMonitor()), \
                 patch('api.inference.image.rank_transport.subprocess.run',return_value=SimpleNamespace(stdout='50\n')) as gpu:
                transport=self.transport()
                if accepted:thermal_guard.check_cpu(root,'test');transport._guard()
                else:
                    with self.assertRaises(RuntimeError):thermal_guard.check_cpu(root,'test')
                    with self.assertRaises(RuntimeError):transport._guard()
                    gpu.assert_not_called()
                record=json.loads((Path(root)/'cpu-thermal.jsonl').read_text())
                self.assertEqual(record['policy'],'unverified-cpu-conservative-75-80-v1')
                self.assertEqual(record['abort_c'],80);self.assertFalse(record['mapping_verified'])

    def test_disappearing_extra_ccd_is_rejected_by_both_owners(self):
        data=sensors();data.append(dict(data[0],label='Tccd1',path='temp3_input'))
        for monitor in (cpu.CpuMonitor(),self.transport().cpu_monitor):
            with patch.object(cpu,'read_identity',return_value=cpu.IDENTITY),patch.object(cpu,'read_cpu_sensors',side_effect=[data,sensors()]):
                self.assertTrue(monitor.sample()['accepted'])
                result=monitor.sample();self.assertFalse(result['accepted']);self.assertIn('sensor_inventory_changed',result['mapping_errors'])

    def test_missing_or_stale_stops_transport_before_gpu_query(self):
        for mode in ('missing','stale','unmapped'):
            data=sensors()
            if mode=='missing':data.pop()
            if mode=='unmapped':data[1]['label']='unknown-hot-sensor';data[1]['celsius']=100
            with patch.object(cpu,'read_cpu_sensors',return_value=data),patch.object(cpu,'read_identity',return_value=cpu.IDENTITY), \
                 patch.object(cpu.time,'monotonic',side_effect=[100,101 if mode=='stale' else 100.01]), \
                 patch('api.inference.image.rank_transport.subprocess.run') as gpu:
                with self.assertRaises(RuntimeError):self.transport()._guard()
                gpu.assert_not_called()

    def test_missing_driver_name_keeps_numeric_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);hw=root/'hwmon1';hw.mkdir();(hw/'temp5_input').write_text('91000')
            rows=cpu.read_cpu_sensors(root)
            self.assertEqual(rows[0]['celsius'],91);self.assertIsNone(rows[0]['label']);self.assertIn('error',rows[0])
            self.assertFalse(cpu.evaluate(rows,cpu.IDENTITY)['accepted'])

    def test_generic_intel_sensors_need_no_amd_pci_metadata(self):
        identity=dict(cpu.IDENTITY,vendor='GenuineIntel',model_name='Unreviewed Intel CPU')
        row=dict(driver='coretemp',label='Package id 0',path='temp1_input',device='coretemp.0',celsius=79)
        self.assertTrue(cpu.evaluate([row],identity)['accepted'])
        row['celsius']=80;self.assertFalse(cpu.evaluate([row],identity)['accepted'])
