import logging
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from api.inference.image.rank_processes import ImageRankProcesses


def event(step=1,job='a'*32,rank=0):
    return f'image_step_returned job={job} rank={rank} step={step} monotonic=1.000000 elapsed_seconds=0.500000\n'.encode()


class ProgressTests(unittest.TestCase):
    def transport(self):
        value=ImageRankProcesses(Path('/unused'),SimpleNamespace());value.session='owned';value.progress_job='a'*32
        return value

    def test_fragmented_records_are_immediate_bounded_and_owned(self):
        value=self.transport()
        with self.assertLogs('api.inference.image.rank_processes',level='INFO') as logs:
            data=event();value._progress(0,data[:20]);self.assertFalse(value.progress_steps)
            value._progress(0,data[20:])
            value._progress(0,event()) # duplicate
            value._progress(0,event(2,job='b'*32)) # wrong job
            value._progress(0,event(2,rank=1)) # wrong rank
            value._progress(0,b'x'*10000+b'\n')
            for n in range(2,42):value._progress(0,event(n))
        self.assertEqual(len(logs.records),40);self.assertEqual(value.progress_steps,{0:40})
        self.assertTrue(all(len(line)<=512 for line in value.progress_lines.values()))

    def test_saved_event_survives_forced_supervisor_kill_without_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.log'
            # No ranks, models or GPUs: a real pipe and log sink in a disposable supervisor.
            code="""
import logging,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from api.inference.image.rank_processes import ImageRankProcesses
r,w=os.pipe();os.set_blocking(r,False)
t=ImageRankProcesses(Path('/unused'),SimpleNamespace());t.session='owned';t.progress_job='a'*32
t.processes=[SimpleNamespace(stderr=os.fdopen(r,'rb',buffering=0))]
logger=logging.getLogger('api.inference.image.rank_processes');logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(sys.argv[1]))
os.write(w,('image_step_returned job='+('a'*32)+' rank=0 step=1 monotonic=1.000000 elapsed_seconds=0.500000\\n').encode())
t._drain_stderr();print('persisted',flush=True);time.sleep(60)
"""
            child=subprocess.Popen([sys.executable,'-u','-c',code,str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(child.stdout,selectors.EVENT_READ)
                    self.assertTrue(selector.select(5));self.assertEqual(child.stdout.readline(),b'persisted\n')
                os.kill(child.pid,signal.SIGKILL);child.wait(timeout=1)
                self.assertIn('image_step_returned session=owned',path.read_text())
            finally:
                if child.poll() is None:child.kill();child.wait(timeout=1)
                child.stdout.close();child.stderr.close()
