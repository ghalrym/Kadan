"""Deterministic fake subprocess/container transport; no Docker or GPU access."""
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest

from native.tests.reference_capture.supervisor import Observation, Policy, supervise

ID='a'*64
CLEAN=Observation(ID,False,0,0,True,True,True,False,0,True)
DIRTY=replace(CLEAN,running=True,process_count=1,child_reaped=False,memory_released=False)


class Clock:
    now=0
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=seconds


class FakeSubprocessStage:
    def __init__(self, *, code=0, observation=CLEAN, failures=(), artifact=True):
        self.code=code;self.observation=observation;self.failures=failures;self.artifact=artifact
        self.commands=[]
    def command(self,name,container_id,timeout):
        self.commands.append((name,container_id,timeout))
        if name in self.failures:raise subprocess.TimeoutExpired([name,container_id],timeout,output=b'partial\xff',stderr=b'error\xfe')
    def verify(self,container_id,timeout):self.command('verify',container_id,timeout);return True
    def start(self,container_id,timeout):self.command('start',container_id,timeout)
    def poll(self,container_id,timeout):self.command('poll',container_id,timeout);return self.code
    def terminate(self,container_id,signal,timeout):self.command(signal,container_id,timeout)
    def observe(self,container_id,timeout):self.command('observe',container_id,timeout);return self.observation
    def validate_artifacts(self,container_id,timeout):self.command('artifact',container_id,timeout);return self.artifact


class SupervisorTests(unittest.TestCase):
    def run_stage(self, stage):
        with tempfile.TemporaryDirectory() as folder:
            clock=Clock()
            report=supervise(stage,ID,Path(folder)/'result.json',execute=True,
                             policy=Policy(deadline=2,term_grace=1,cleanup_deadline=6),clock=clock,sleep=clock.sleep)
            self.assertTrue(all(call[1]==ID and 0<call[2]<=5 for call in stage.commands))
            self.assertLessEqual(clock.now,9)
            return report

    def test_success_requires_five_observations(self):
        stage=FakeSubprocessStage();report=self.run_stage(stage)
        self.assertTrue(report['accepted']);self.assertTrue(report['restoration_permitted'])
        self.assertGreaterEqual(sum(c[0]=='observe' for c in stage.commands),5)

    def test_deadline_kill_is_not_exit_proof(self):
        stage=FakeSubprocessStage(code=None,observation=DIRTY)
        report=self.run_stage(stage)
        self.assertFalse(report['accepted']);self.assertFalse(report['restoration_permitted'])
        self.assertIn('KILL_requested_not_proof_of_exit',report['events'])

    def test_failed_term_kill_wait_and_start_hold_ownership(self):
        for failures in (('TERM','KILL'),('observe',),('start',)):
            stage=FakeSubprocessStage(observation=DIRTY,failures=failures)
            report=self.run_stage(stage)
            self.assertFalse(report['restoration_permitted'])
            self.assertIn('cleanup_uncertain_hold_ownership_no_restoration',report['errors'])

    def test_timeout_can_restore_only_after_observed_cleanup_but_cannot_pass(self):
        report=self.run_stage(FakeSubprocessStage(code=None))
        self.assertFalse(report['accepted']);self.assertTrue(report['cleanup_verified'])

    def test_oom_missing_done_drift_nonzero_fail_even_with_clean_exit(self):
        for stage in (FakeSubprocessStage(observation=replace(CLEAN,docker_oom=True)),
                      FakeSubprocessStage(observation=replace(CLEAN,cgroup_oom_kill_delta=1)),
                      FakeSubprocessStage(observation=replace(CLEAN,manifest_unchanged=False)),
                      FakeSubprocessStage(artifact=False),FakeSubprocessStage(code=1)):
            report=self.run_stage(stage)
            self.assertFalse(report['accepted']);self.assertTrue(report['cleanup_verified'])

    def test_unknown_or_wrong_identity_never_proves_cleanup(self):
        for change in ({'container_id':'b'*64},{'running':None},{'process_count':False},
                       {'compute_process_count':None},{'child_reaped':None},{'memory_released':None},
                       {'gpu_baseline_restored':None},{'cgroup_oom_kill_delta':None}):
            report=self.run_stage(FakeSubprocessStage(observation=replace(CLEAN,**change)))
            self.assertFalse(report['restoration_permitted'])

    def test_inert_and_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'result';p.write_bytes(b'preserve');stage=FakeSubprocessStage()
            with self.assertRaises(ValueError):supervise(stage,ID,p)
            with self.assertRaises(ValueError):supervise(stage,'mutable-name',p,execute=True)
            with self.assertRaises(FileExistsError):supervise(stage,ID,p,execute=True)
            self.assertEqual(stage.commands,[]);self.assertEqual(p.read_bytes(),b'preserve')


if __name__=='__main__':unittest.main()
