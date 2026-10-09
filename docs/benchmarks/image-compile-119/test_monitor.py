import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('compile_monitor',Path(__file__).with_name('monitor.py'))
monitor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def row(self):
        return dict(host_available_bytes=monitor.CONFIG['physical_host_start_min_bytes'],
            gpus={monitor.UUID:dict(free_mib=monitor.CONFIG['gpu_budget_bytes']/1024**2,
                used_mib=0,power_brake='Not Active')})

    def test_thermal_flag_does_not_abort_admission_or_running_work(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(monitor,'ROOT',Path(directory)):
            for thermal in ('Active','Not Active','N/A',None):
                for initial in (False,True):
                    with self.subTest(thermal=thermal,initial=initial):
                        row=self.row();row['gpus'][monitor.UUID]['thermal']=thermal
                        self.assertIsNone(monitor.reason(row,initial))
            self.assertIsNone(monitor.reason(self.row()))

    def test_sample_retains_passive_thermal_flag(self):
        output=f'{monitor.UUID}, 24000, 100, Active, Not Active\n'
        with patch.object(monitor,'command',return_value=output),patch.object(Path,'read_text',return_value='MemAvailable: 1000000 kB\n'):
            row=monitor.sample()
        self.assertEqual(row['gpus'][monitor.UUID]['thermal'],'Active')
        self.assertEqual(row['gpus'][monitor.UUID]['power_brake'],'Not Active')

    def test_resource_failures_still_abort_with_thermal_flag_active(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(monitor,'ROOT',Path(directory)):
            for fault,expected in [('host','host RAM headroom'),('gpu','GPU1 physical headroom'),('cgroup','cgroup RAM safety margin')]:
                row=self.row();row['gpus'][monitor.UUID]['thermal']='Active'
                if fault=='host':row['host_available_bytes']=0
                if fault=='gpu':row['gpus'][monitor.UUID]['free_mib']=0
                if fault=='cgroup':row['cgroup']={'current':monitor.CONFIG['host_budget_bytes']}
                self.assertEqual(monitor.reason(row),expected)

    def test_existing_power_brake_decision_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(monitor,'ROOT',Path(directory)):
            row=self.row();row['gpus'][monitor.UUID]['power_brake']='Active'
            self.assertEqual(monitor.reason(row),'GPU1 power-brake flag')


if __name__=='__main__':unittest.main()
