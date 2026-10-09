import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from durable_evidence import ImageEvents, append
from local_evidence_run import command


class DurableEvidenceTests(unittest.TestCase):
    def test_completed_record_survives_producer_kill(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'events.jsonl'
            code="from durable_evidence import append;import sys,time;append(sys.argv[1],{'done':True});print('saved',flush=True);time.sleep(30)"
            process=subprocess.Popen([sys.executable,'-c',code,str(path)],stdout=subprocess.PIPE,text=True,
                env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent)))
            try:
                self.assertEqual(process.stdout.readline(),'saved\n')
                process.kill();process.wait(timeout=5)
                self.assertEqual(json.loads(path.read_text()),{'done':True})
            finally:
                if process.poll() is None:process.kill();process.wait()
                process.stdout.close()

    def test_caps_preserve_prior_record(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'events'
            append(path,{'a':1},cap=8);before=path.read_bytes()
            with self.assertRaises(ValueError):append(path,{'b':2},cap=8)
            with self.assertRaises(ValueError):append(path,{'big':'x'*10},record_cap=8)
            self.assertEqual(path.read_bytes(),before)

    def test_no_symlink_following(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'target';target.write_text('keep')
            link=Path(root)/'link';link.symlink_to(target)
            with self.assertRaises(OSError):append(link,{'bad':True})
            self.assertEqual(target.read_text(),'keep')

    def test_only_selected_events(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'events';handler=ImageEvents(path)
            for message in ('unrelated prompt','image_step_returned {}','dual_image_timing {}','dual_image_published {}'):
                handler.handle(logging.LogRecord('api.test',logging.INFO,'',0,message,(),None))
            self.assertEqual(len(path.read_text().splitlines()),3)
            self.assertNotIn('unrelated',path.read_text())

    def test_local_manager_lifetime_and_no_restart(self):
        with tempfile.TemporaryDirectory() as root:
            argv=command('kadan-reviewed-cpu-test',root,['/usr/bin/python3','script with spaces.py'])
            self.assertIn('--property=Restart=no',argv)
            self.assertIn('--property=RuntimeMaxSec=900',argv)
            self.assertIn('--property=KillMode=mixed',argv)
            self.assertNotIn('--scope',argv);self.assertNotIn('--wait',argv)
            self.assertEqual(argv[-2:],['/usr/bin/python3','script with spaces.py'])
            with self.assertRaises(ValueError):command('bad/name',root,['/bin/true'])
            with self.assertRaises(ValueError):command('kadan-reviewed-x',root,['relative'])


if __name__=='__main__':unittest.main()
