from dataclasses import replace
import unittest
from .gpu_gate import Snapshot,GpuGate,UUID,ENTRY_FREE,DEVICE_BUDGET,parse_snapshot

OTHER='GPU-other'
CID='a'*64


class GpuGateTests(unittest.TestCase):
    def baseline(self):
        return Snapshot({UUID:{'index':0,'free_bytes':24000*1024**2,'temperature':50},
                         OTHER:{'index':1,'free_bytes':20000*1024**2,'temperature':52}},(),False)

    def test_admission_and_cleanup(self):
        base=self.baseline();gate=GpuGate(base)
        self.assertTrue(gate.evaluate(base,CID,cleanup=True))
        for changes in ({'free_bytes':ENTRY_FREE-1},{'temperature':66}):
            bad=self.baseline();bad.devices[UUID].update(changes)
            with self.assertRaises(ValueError):GpuGate(bad)
        with self.assertRaises(ValueError):GpuGate(replace(base,xid=True))

    def test_ownership_thermal_memory_and_xid_failures(self):
        gate=GpuGate(self.baseline())
        for fault in ('foreign','over_budget','hot','low_free','xid','other','cleanup_resident','cleanup_bytes'):
            sample=self.baseline();owned={'uuid':UUID,'pid':42,'used_bytes':DEVICE_BUDGET,'cgroup':f'0::/system.slice/docker-{CID}.scope'}
            if fault=='foreign':owned['cgroup']='0::/unrelated'
            if fault=='over_budget':owned['used_bytes']=DEVICE_BUDGET+64*1024**2+1
            if fault in ('foreign','over_budget','cleanup_resident'):sample=replace(sample,processes=(owned,))
            if fault=='hot':sample.devices[OTHER]['temperature']=80
            if fault=='low_free':sample.devices[UUID]['free_bytes']=1024**3-1
            if fault=='xid':sample=replace(sample,xid=True)
            if fault=='other':sample=replace(sample,processes=({'uuid':OTHER,'pid':8,'used_bytes':1,'cgroup':'0::/new'},))
            if fault=='cleanup_bytes':sample.devices[UUID]['free_bytes']-=65*1024**2
            with self.subTest(fault=fault),self.assertRaises(ValueError):
                gate.evaluate(sample,CID,cleanup=fault.startswith('cleanup'))

    def test_parser_rejects_missing_unknown_telemetry(self):
        gpu=f'0, {UUID}, 24000, 50\n'
        self.assertEqual(parse_snapshot(gpu,'',{},'').devices[UUID]['index'],0)
        for processes,groups in ((f'{UUID}, 42, 1\n',{}),(f'{UUID}, 42, N/A\n',{'42':'0::/x'})):
            with self.assertRaises(ValueError):parse_snapshot(gpu,processes,groups,'')
        with self.assertRaises(ValueError):parse_snapshot(gpu.replace('24000','N/A'),'',{},'')


if __name__=='__main__':unittest.main()
