import ctypes
import json
import sys
import time
import threading
import subprocess
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from nvml_sensor import Memory, NVML, NVMLReader, parent_fence

UUID='GPU-e30b6419-2c6d-f550-61d6-16166a920dac'


def fake_library(*, identity=UUID, status=0):
    def init():return 0
    def handle(uuid,pointer):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_void_p))[0]=ctypes.c_void_p(1);return 0
    def identity_fn(handle,buffer,size):ctypes.memmove(buffer,identity.encode()+b'\0',len(identity)+1);return 0
    def memory(handle,pointer):
        value=ctypes.cast(pointer,ctypes.POINTER(Memory)).contents
        assert value.version==ctypes.sizeof(Memory)|(2<<24)
        value.total=1000*1048576;value.used=400*1048576;value.reserved=100*1048576;value.free=600*1048576
        return status
    def temperature(handle,sensor,pointer):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_uint))[0]=65;return 0
    return SimpleNamespace(nvmlInit_v2=init,nvmlShutdown=init,nvmlDeviceGetHandleByUUID=handle,
        nvmlDeviceGetUUID=identity_fn,nvmlDeviceGetMemoryInfo_v2=memory,nvmlDeviceGetTemperature=temperature)


class NVMLTests(unittest.TestCase):
    def test_typed_uuid_bound_memory_and_timing(self):
        sensor=NVML([UUID],fake_library());reply=sensor.sample()
        self.assertEqual(reply['gpu'][UUID],dict(used=400,reserved=100,free=600,temperature=65))
        self.assertEqual(set(reply['nvml_call_seconds'][UUID]),{'memory_seconds','temperature_seconds'})

    def test_identity_duplicate_and_driver_errors_fail_closed(self):
        for ids,library in [([UUID],fake_library(identity='wrong')),([UUID,UUID],fake_library()),([],fake_library())]:
            with self.assertRaises(ValueError):NVML(ids,library)
        with self.assertRaisesRegex(RuntimeError,'NVML status 999'):NVML([UUID],fake_library(status=999)).sample()

    def test_parent_death_fence_rejects_lost_owner(self):
        library=SimpleNamespace(prctl=Mock(return_value=0))
        with patch('nvml_sensor.ctypes.CDLL',return_value=library),patch('nvml_sensor.os.getppid',return_value=123):
            parent_fence(123)
            with self.assertRaises(RuntimeError):parent_fence(124)
        self.assertEqual(library.prctl.call_count,2)

    def reader(self,code):
        reader=NVMLReader([UUID],command=[sys.executable,'-u','-c',code]);self.addCleanup(reader.close)
        return reader

    def test_one_process_reused_and_reaped(self):
        reader=self.reader("import sys,json\nfor line in sys.stdin: print(json.dumps({'sequence':int(line),'gpu':{}}),flush=True)")
        reader.sample(timeout=1);pid=reader.process.pid
        reply=reader.sample(timeout=1);self.assertEqual(reader.process.pid,pid)
        self.assertEqual(reply['spawn_to_exec_seconds'],0)
        reader.close();reader.close();self.assertIsNotNone(reader.process.poll())

    def test_close_during_blocked_popen_reaps_before_any_request(self):
        reader=self.reader('import time;time.sleep(60)')
        entered=threading.Event();release=threading.Event();children=[];errors=[]
        popen=subprocess.Popen
        def blocked(*args,**kwargs):
            entered.set()
            if not release.wait(2):raise RuntimeError('test release timeout')
            child=popen(*args,**kwargs)
            child.stdin=Mock(wraps=child.stdin)
            children.append(child)
            return child
        def sample():
            try:reader.sample(timeout=1)
            except Exception as exc:errors.append(exc)
        with patch('nvml_sensor.subprocess.Popen',side_effect=blocked):
            worker=threading.Thread(target=sample)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                started=time.monotonic();reader.close()
                self.assertLess(time.monotonic()-started,.2)
                self.assertIsNone(reader.process)
                release.set();worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(children),1)
                children[0].stdin.write.assert_not_called()
                self.assertIsNotNone(children[0].poll())
                self.assertTrue(children[0].stdout.closed)
                self.assertEqual(len(errors),1)
                self.assertRegex(str(errors[0]),'closed during startup')
                self.assertIsNone(reader.process)
                with self.assertRaisesRegex(RuntimeError,'closed'):reader.sample(timeout=1)
            finally:
                release.set();worker.join(3)
                for child in children:
                    if child.poll() is None:child.kill();child.wait(timeout=1)
                    child.stdin.close();child.stdout.close()

    def test_hung_driver_helper_has_bounded_timeout_and_cleanup(self):
        reader=self.reader('import time;time.sleep(60)')
        with self.assertRaises(TimeoutError):reader.sample(timeout=.1)
        started=time.monotonic();reader.close()
        self.assertLess(time.monotonic()-started,.5);self.assertIsNotNone(reader.process.poll())

    def test_exit_bad_sequence_oversize_and_error_fail_closed(self):
        codes=["pass", "print('not-json',flush=True)",
               "print('{\"sequence\":99}',flush=True)","print('x'*9000,flush=True)",
               "print('{\"sequence\":1,\"error\":\"driver lost\"}',flush=True)"]
        for code in codes:
            with self.subTest(code=code):
                reader=self.reader(code)
                with self.assertRaises((RuntimeError,ValueError,BrokenPipeError)):reader.sample(timeout=1)
                reader.close();self.assertIsNotNone(reader.process.poll())


if __name__=='__main__':unittest.main()
