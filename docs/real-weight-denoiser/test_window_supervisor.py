import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_evidence_run import command
from window_supervisor import context, digest, recover, supervise, LABEL, OWNER_GRACE, require_launch_budget, DockerOps, recovery_action, write_once


class Clock:
    def __init__(self,value=0):self.value=value
    def now(self):return self.value
    def sleep(self,seconds):self.value+=seconds


class Docker:
    def __init__(self,clock,ready_seconds=180):
        self.clock=clock;self.ready_seconds=ready_seconds;self.actions=[];self.clear=True
        self.original=dict(Id='a'*64,Image='image',Config={},HostConfig={},Mounts=[],Path='python',Args=['-m','api'],State={'Running':False})
        self.temporary=dict(Id='b'*64,Config={'Labels':{LABEL:'token'}})
    def inspect(self,name):
        self.clock.sleep(3);self.actions.append(('inspect',name))
        if name in ('kadan-api-dual-reviewed','b'*64):return self.temporary
        if name in ('kadan-api-1','a'*64):return self.original
        return None
    def source_matches(self,record):self.clock.sleep(6);return True
    def remove(self,name):
        assert name=='b'*64;self.clock.sleep(15);self.actions.append(('remove',name));self.temporary=None
    def physical_clear(self,desktop):
        self.clock.sleep(30);self.actions.append(('physical',None))
        if not self.clear:raise RuntimeError('QUARANTINE: physical owners remain')
    def start(self,name):
        self.clock.sleep(45);self.actions.append(('start',name));self.original['State']['Running']=True
    def ready(self,name):
        self.clock.sleep(self.ready_seconds);self.actions.append(('ready',name))


class Child:
    def __init__(self,clock,*,cleanup=None,finish=None):
        self.clock=clock;self.cleanup=cleanup;self.finish=finish;self.returncode=None;self.terms=[];self.killed=False
    def poll(self):
        if self.finish is not None and self.clock.now()>=self.finish:self.returncode=0
        return self.returncode
    def terminate(self):
        self.terms.append(self.clock.now())
        if self.cleanup is not None:self.finish=self.clock.now()+self.cleanup
    def kill(self):self.killed=True;self.returncode=-9
    def wait(self,timeout):assert self.killed;return self.returncode


def fixture(clock,ready=180):
    ctx=context(0,'token','boot');ops=Docker(clock,ready)
    record=dict(token='token',end=ctx['end'],api_id='a'*64,identity=digest(ops.original),others={},desktop=[],temporary_name='kadan-api-dual-reviewed')
    return ctx,ops,record


class WindowSupervisorTests(unittest.TestCase):
    def test_late_service_activation_cannot_begin_a_window(self):
        ctx=context(0,'token','boot');require_launch_budget(ctx,15)
        with self.assertRaises(TimeoutError):require_launch_budget(ctx,15.001)

    def test_service_has_separate_stop_post_and_sufficient_grace(self):
        with tempfile.TemporaryDirectory() as directory:
            args=command('kadan-reviewed-test',directory,['/usr/bin/python3','operator.py'],started=100)
            ctx=json.loads((Path(directory)/'window.json').read_text())
            self.assertEqual(ctx['work_deadline'],1000);self.assertEqual(ctx['operator_end'],1450);self.assertEqual(ctx['end'],1900)
            self.assertIn('--property=RuntimeMaxSec=900',args);self.assertIn('--property=TimeoutStopSec=450',args)
            post=next(a for a in args if a.startswith('--property=ExecStopPost='))
            self.assertIn('--recover',post);self.assertIn(directory,post)
            self.assertIn('--property=KillMode=mixed',args);self.assertIn('--property=Restart=no',args)
            with self.assertRaises(FileExistsError):command('kadan-reviewed-test',directory,['/bin/true'])

    def test_preflight_uses_the_same_clock_and_stop_allows_180s_restore(self):
        for restoration in (75,180):
            clock=Clock(400);ctx,ops,record=fixture(clock,0);child=Child(clock,cleanup=restoration)
            supervise(child,ctx,stopping=lambda:False,clock=clock.now,sleep=clock.sleep)
            self.assertLess(abs(child.terms[0]-900),.11)
            self.assertFalse(child.killed);self.assertEqual(len(child.terms),1)
            # Model the operator's successful cleanup before service exit.
            ops.temporary=None;ops.original['State']['Running']=True
            self.assertTrue(recover(ctx,record,ops)['restored']);self.assertLess(clock.now(),ctx['operator_end'])

    def test_hung_operator_is_killed_then_outside_docker_is_removed(self):
        clock=Clock();ctx,ops,record=fixture(clock);child=Child(clock)
        supervise(child,ctx,stopping=lambda:False,clock=clock.now,sleep=clock.sleep)
        self.assertTrue(child.killed);self.assertIsNotNone(ops.temporary)
        self.assertTrue(recover(ctx,record,ops)['restored']);self.assertIsNone(ops.temporary)
        actions=[name for name,_ in ops.actions]
        self.assertLess(actions.index('remove'),actions.index('physical'));self.assertLess(actions.index('physical'),actions.index('start'))
        self.assertLess(clock.now(),ctx['end'])

    def test_supervisor_hard_timeout_still_has_independent_stop_post_budget(self):
        # Even if the supervisor never handles TERM, manager kills its cgroup
        # after450s. Docker is outside that cgroup; ExecStopPost does its cleanup.
        clock=Clock(1350+15);ctx,ops,record=fixture(clock)
        self.assertTrue(recover(ctx,record,ops)['restored'])
        self.assertIsNone(ops.temporary);self.assertLess(clock.now(),ctx['end'])

    def test_manual_stop_is_sent_once_and_keeps_restore_reserve(self):
        clock=Clock();ctx,ops,record=fixture(clock);child=Child(clock,cleanup=180)
        supervise(child,ctx,stopping=lambda:clock.now()>=20,clock=clock.now,sleep=clock.sleep)
        self.assertEqual(len(child.terms),1);self.assertFalse(child.killed)
        self.assertTrue(recover(ctx,record,ops)['restored']);self.assertLess(clock.now(),20+450+450)

    def test_pre_pause_death_has_no_docker_action(self):
        clock=Clock();ctx,ops,_=fixture(clock)
        self.assertEqual(recover(ctx,None,ops),dict(armed=False,restored=False));self.assertEqual(ops.actions,[])

    def test_foreign_label_is_never_removed(self):
        clock=Clock();ctx,ops,record=fixture(clock);ops.temporary['Config']['Labels'][LABEL]='foreign'
        with self.assertRaisesRegex(RuntimeError,'another owner'):recover(ctx,record,ops)
        self.assertFalse(any(name in ('remove','start') for name,_ in ops.actions))

    def test_failed_physical_cleanup_does_not_start_api(self):
        clock=Clock();ctx,ops,record=fixture(clock);ops.clear=False
        with self.assertRaisesRegex(RuntimeError,'physical owners'):recover(ctx,record,ops)
        self.assertIsNone(ops.temporary);self.assertFalse(ops.original['State']['Running'])

    def test_source_change_prevents_original_restart(self):
        clock=Clock();ctx,ops,record=fixture(clock)
        ops.source_matches=lambda record:False
        with self.assertRaisesRegex(RuntimeError,'source changed'):recover(ctx,record,ops)
        self.assertIsNone(ops.temporary)
        self.assertFalse(any(name=='start' for name,_ in ops.actions))

    def test_original_identity_change_is_quarantined(self):
        clock=Clock();ctx,ops,record=fixture(clock)
        ops.original['Image']='changed'
        with self.assertRaisesRegex(RuntimeError,'identity changed'):recover(ctx,record,ops)
        self.assertFalse(any(name=='start' for name,_ in ops.actions))

    def test_command_timeout_uses_absolute_and_phase_deadlines(self):
        ops=DockerOps(1800)
        with patch('window_supervisor.time.monotonic',return_value=1799):
            self.assertEqual(ops.remaining(45),1)
            self.assertEqual(ops.remaining(3,1799.5),.5)
        with patch('window_supervisor.time.monotonic',return_value=1800):
            with self.assertRaises(TimeoutError):ops.remaining(3)

    def test_failed_recovery_is_durably_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            write_once(Path(directory)/'recovery.json',{'token':'wrong'})
            with patch('window_supervisor.recover',side_effect=RuntimeError('QUARANTINE: fixture')):
                with self.assertRaisesRegex(RuntimeError,'QUARANTINE'):
                    recovery_action(directory,{'end':1800})
            event=json.loads((Path(directory)/'recovery-events.jsonl').read_text())
            self.assertFalse(event['restored']);self.assertIn('QUARANTINE',event['error'])

    def test_stop_signal_during_stop_post_does_not_interrupt_restore(self):
        code="""import os,signal
from test_window_supervisor import Clock,fixture
from window_supervisor import recover
flag=[]
signal.signal(signal.SIGTERM,lambda *args:flag.append(True))
clock=Clock(1350);ctx,ops,record=fixture(clock)
ready=ops.ready
def interrupted_ready(name):
    os.kill(os.getpid(),signal.SIGTERM)
    ready(name)
ops.ready=interrupted_ready
assert recover(ctx,record,ops)['restored'] and flag and clock.now()<ctx['end']
"""
        subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,check=True,timeout=5)


if __name__=='__main__':unittest.main()
