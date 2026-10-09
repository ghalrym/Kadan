import json
from pathlib import Path
import tempfile
import unittest

from thermal_guard import check_cpu, read_cpu_sensors
from thermal_policy import IDENTITY, evaluate


def rows(value=50):
    return [dict(driver='k10temp',label=label,path='/sys/class/hwmon/hwmon3/'+name,
                 celsius=value,pci_vendor='0x1022',pci_device='0x1653',device='0000:00:18.3')
            for label,name in [('Tctl','temp1_input'),('Tccd3','temp5_input'),('Tccd5','temp7_input')]]


class ThermalPolicyTests(unittest.TestCase):
    def test_exact_boundaries_on_every_sensor(self):
        for index in range(3):
            for value,state,accepted in [(84.999,'normal',True),(85,'warning',True),(89.999,'warning',True),(90,'abort',False),(90.001,'abort',False)]:
                with self.subTest(sensor=index,value=value):
                    data=rows();data[index]['celsius']=value;r=evaluate(data,IDENTITY)
                    self.assertEqual(r['assessments'][index]['state'],state);self.assertEqual(r['accepted'],accepted)

    def test_main_sensor_does_not_hide_hot_ccd(self):
        data=rows();data[0]['celsius']=69.125;data[1]['celsius']=90
        self.assertFalse(evaluate(data,IDENTITY)['accepted'])
        data[1]['celsius']=80.75;self.assertTrue(evaluate(data,IDENTITY)['accepted'])

    def test_every_expected_sensor_required(self):
        for index in range(3):
            data=rows();del data[index];self.assertIn('missing_required_sensor',evaluate(data,IDENTITY)['mapping_errors'])
        self.assertFalse(evaluate([],IDENTITY)['accepted'])

    def test_unknown_duplicate_or_unreadable_is_not_dropped(self):
        for change in ({'label':'mystery'},{'path':'temp9_input'},{'driver':'coretemp'},
                       {'pci_device':'0x0000'},{'celsius':float('nan')},{'celsius':float('inf')},
                       {'celsius':None},{'error':'gone'}):
            data=rows();data[1].update(change);result=evaluate(data,IDENTITY)
            self.assertFalse(result['accepted']);self.assertEqual(len(result['assessments']),3)
        data=rows();data.append(data[0].copy());self.assertFalse(evaluate(data,IDENTITY)['accepted'])
        # A newly exposed, valid CCD must also be checked, never ignored.
        data=rows();data.append(dict(data[1],label='Tccd1',path='temp3_input',celsius=91))
        self.assertEqual(evaluate(data,IDENTITY)['assessments'][-1]['state'],'abort')

    def test_changed_cpu_module_or_kernel_requires_review(self):
        for key in IDENTITY:
            result=evaluate(rows(80),dict(IDENTITY,**{key:'different'}))
            self.assertFalse(result['accepted']);self.assertEqual(result['abort_c'],80)
            self.assertFalse(result['mapping_verified'])

    def test_warning_logged_and_all_readings_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            data=rows();data[2]['celsius']=85
            with self.assertLogs('thermal_guard',level='WARNING') as log:
                self.assertEqual(check_cpu(directory,'run',data,identity=IDENTITY),85)
            self.assertIn('abort=90',log.output[0])
            sample=json.loads((Path(directory)/'cpu-thermal.jsonl').read_text())
            self.assertEqual(sample['sensors'],data);self.assertTrue(sample['warning']);self.assertTrue(sample['accepted'])
            data[2]['celsius']=90
            with self.assertRaises(RuntimeError):check_cpu(directory,'run',data,identity=IDENTITY)
            sample=json.loads((Path(directory)/'cpu-thermal.jsonl').read_text().splitlines()[-1])
            self.assertFalse(sample['accepted']);self.assertEqual(sample['peak_c'],90)

    def test_cool_admission_and_no_relaxation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(check_cpu(directory,'admission',rows(59.999),limit=60,identity=IDENTITY),59.999)
            with self.assertRaises(RuntimeError):check_cpu(directory,'admission',rows(60),limit=60,identity=IDENTITY)
            for limit in (90.001,float('nan'),float('inf'),0):
                with self.assertRaises(ValueError):check_cpu(directory,'run',rows(),limit=limit,identity=IDENTITY)

    def test_unreadable_label_retains_sensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);hw=root/'hwmon3';hw.mkdir()
            (hw/'name').write_text('k10temp');(hw/'temp5_input').write_text('91000')
            data=read_cpu_sensors(root)
            self.assertEqual(len(data),1);self.assertIn('error',data[0]);self.assertEqual(data[0]['celsius'],91)
            self.assertFalse(evaluate(data,IDENTITY)['accepted'])


if __name__=='__main__':unittest.main()
