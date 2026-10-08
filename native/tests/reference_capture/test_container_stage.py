"""Transport tests replace every subprocess; no container operations occur."""
from pathlib import Path
import subprocess
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from dataclasses import replace
from native.tests.reference_capture.ancestor_observer import AncestorObserver, HeadroomProof, identity
from native.tests.reference_capture.supervisor import Observation

from native.tests.reference_capture.container_stage import DockerReferenceStage
from native.tests.reference_capture.artifacts import sha256
from native.tests.reference_capture.evidence import read_regular


class TransportTests(unittest.TestCase):
    def bare_stage(self):
        stage=object.__new__(DockerReferenceStage)
        stage.verified_id='a'*64
        stage.child=Mock()
        stage.logs=[]
        stage.evidence_children=[]
        stage.evidence_check=Mock(return_value=True)
        return stage

    def test_failed_kill_rpc_does_not_manufacture_exit(self):
        stage=self.bare_stage()
        with patch('native.tests.reference_capture.container_stage.subprocess.run',
                   side_effect=subprocess.TimeoutExpired(['docker','kill'],1)),self.assertRaises(subprocess.TimeoutExpired):
            stage.terminate(stage.verified_id,'KILL',1)
        stage.child.wait.assert_not_called()

    def test_rpc_output_is_disk_backed_and_timeout_bounded(self):
        stage=self.bare_stage()
        with patch('native.tests.reference_capture.container_stage.subprocess.run') as run:
            run.return_value.returncode=0
            self.assertEqual(stage.rpc(['inspect',stage.verified_id],2),b'')
            self.assertEqual(run.call_args.kwargs['timeout'],2)
            self.assertNotIn('capture_output',run.call_args.kwargs)
            self.assertEqual(run.call_args.args[0][:2],['docker','inspect'])

    def test_missing_cgroup_or_unreaped_client_blocks_cleanup(self):
        for running in (False,True):
            stage=self.bare_stage()
            with tempfile.TemporaryDirectory() as folder:
                stage.cgroup=Path(folder)/'vanished'
                stage.manifest_path=Path(folder)/'manifest';stage.manifest_path.write_bytes(b'{}')
                stage.manifest_digest='0'*64;stage.final_oom=None
                stage.evidence_check.return_value=False
                stage.inspect=Mock(return_value={'State':{'Running':running,'Pid':0,'OOMKilled':False}})
                stage.child.poll.return_value=None
                stage.child.wait.side_effect=subprocess.TimeoutExpired(['docker','exec'],1)
                result=stage.observe(stage.verified_id,1)
                self.assertFalse(result.cleanup_known(stage.verified_id))
                self.assertIsNone(result.cgroup_oom_kill_delta)
                self.assertFalse(result.manifest_unchanged)

    def stopped_stage(self, folder, memory='0', populated='0'):
        stage=self.bare_stage()
        stage.cgroup=Path(folder)/'synthetic-cgroup';stage.cgroup.mkdir()
        for name,value in {'cgroup.procs':'', 'memory.current':memory,
                           'cgroup.events':'populated '+populated+'\n',
                           'memory.events':'oom_kill 0\n'}.items():
            (stage.cgroup/name).write_text(value)
        st=stage.cgroup.stat();stage.cgroup_identity=(st.st_dev,st.st_ino)
        stage.initial_oom=0;stage.final_oom=None
        stage.manifest_path=Path(folder)/'manifest';stage.manifest_path.write_bytes(b'{}')
        stage.manifest_digest=sha256(b'{}')
        stage.inspect=Mock(return_value={'State':{'Running':False,'Pid':0,'OOMKilled':False}})
        stage.child.poll.return_value=0
        return stage

    def test_empty_processes_do_not_prove_charged_memory_release(self):
        for charge in ('1','4096',str(40*1024**3)):
            with self.subTest(charge=charge),tempfile.TemporaryDirectory() as folder:
                stage=self.stopped_stage(folder,memory=charge)
                result=stage.observe(stage.verified_id,1)
                self.assertEqual(result.process_count,0)
                self.assertEqual(result.memory_current_bytes,int(charge))
                self.assertFalse(result.memory_released)
                self.assertFalse(result.cleanup_known(stage.verified_id))

    def test_descendant_population_blocks_cleanup_with_empty_direct_procs(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=self.stopped_stage(folder,populated='1')
            result=stage.observe(stage.verified_id,1)
            self.assertEqual(result.process_count,0)
            self.assertEqual(result.cgroup_populated,1)
            self.assertFalse(result.cleanup_known(stage.verified_id))

    def test_zero_hierarchical_charge_and_population_can_prove_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=self.stopped_stage(folder)
            result=stage.observe(stage.verified_id,1)
            self.assertTrue(result.cleanup_known(stage.verified_id))
            self.assertEqual(result.memory_current_bytes,0)
            self.assertEqual(result.cgroup_populated,0)

    def test_missing_malformed_unreadable_evidence_never_reuses_clean_sample(self):
        for filename in ('memory.current','cgroup.events','memory.events'):
            for fault in ('missing','malformed','unreadable'):
                with self.subTest(filename=filename,fault=fault),tempfile.TemporaryDirectory() as folder:
                    stage=self.stopped_stage(folder)
                    self.assertTrue(stage.observe(stage.verified_id,1).cleanup_known(stage.verified_id))
                    target=stage.cgroup/filename
                    if fault=='missing':target.unlink()
                    elif fault=='malformed':target.write_text('unknown')
                    real_read=Path.read_text
                    def read(path,*args,**kwargs):
                        if fault=='unreadable' and path==target:raise PermissionError('synthetic denial')
                        return real_read(path,*args,**kwargs)
                    with patch.object(Path,'read_text',read),self.assertRaises((OSError,ValueError,KeyError)):
                        stage.observe(stage.verified_id,1)
                    self.assertIsNone(stage.final_oom)

    def test_cgroup_replacement_is_not_cleanup_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            stage=self.stopped_stage(folder)
            stage.cgroup_identity=(-1,-1)
            with self.assertRaisesRegex(ValueError,'cgroup_replaced'):
                stage.observe(stage.verified_id,1)

    def test_evidence_regular_file_bounds_and_no_follow(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'file';p.write_bytes(b'{}')
            self.assertEqual(read_regular(p,2),b'{}')
            p.write_bytes(b'123')
            with self.assertRaisesRegex(ValueError,'evidence_size'):read_regular(p,2)
            link=Path(folder)/'link';link.symlink_to(p)
            with self.assertRaises(OSError):read_regular(link,10)
            fifo=Path(folder)/'fifo';os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError,'evidence_regular_file'):read_regular(fifo,10)
            directory=Path(folder)/'dir';directory.mkdir()
            (directory/'file').write_bytes(b'{}')
            alias=Path(folder)/'alias';alias.symlink_to(directory,target_is_directory=True)
            with self.assertRaises(OSError):read_regular(alias/'file',10)

    def test_evidence_deadline_retains_unreaped_helper_ownership(self):
        stage=self.bare_stage();stage.manifest_digest='0'*64
        # Invoke real transport method; replace only the spawned subprocess.
        helper=Mock();helper.wait.side_effect=subprocess.TimeoutExpired(['helper'],1)
        helper.poll.return_value=None
        with patch('native.tests.reference_capture.container_stage.subprocess.Popen',return_value=helper), \
             self.assertRaises(subprocess.TimeoutExpired):
            DockerReferenceStage.evidence_check(stage,'manifest',Path('/unused'),1)
        helper.kill.assert_called_once()
        helper.wait.assert_called_once()
        self.assertGreater(helper.wait.call_args.kwargs['timeout'],0)
        self.assertLessEqual(helper.wait.call_args.kwargs['timeout'],1)
        self.assertEqual(stage.evidence_children,[helper])
        with tempfile.TemporaryDirectory() as folder:
            observed=self.stopped_stage(folder)
            observed.evidence_children=[helper]
            result=observed.observe(observed.verified_id,1)
            self.assertFalse(result.child_reaped)
            self.assertFalse(result.cleanup_known(observed.verified_id))

    def test_headroom_proof_is_distinct_from_zero_freed_bytes(self):
        proof=HeadroomProof(True,True,True,True,True,137,100*1024**3,44*1024**3,
                            60*1024**3,None,44*1024**3)
        observation=Observation('a'*64,False,0,0,True,True,False,False,0,True,
                                None,None,'terminated_headroom',proof)
        self.assertTrue(observation.cleanup_known('a'*64))
        self.assertFalse(observation.memory_released)
        for key,value in (('init_terminated',False),('absence_verified',False),
                          ('ancestor_identity_unchanged',False),('ancestor_oom_unchanged',False),
                          ('ancestor_peers_unchanged',False),('host_available_bytes',1),
                          ('ancestor_limit_bytes',61*1024**3),('host_floor_bytes',0)):
            with self.subTest(key=key):
                self.assertFalse(replace(observation,headroom_proof=replace(proof,**{key:value})).cleanup_known('a'*64))
        self.assertFalse(replace(observation,memory_released=True).cleanup_known('a'*64))
        self.assertFalse(replace(observation,cleanup_mode='zero_charge').cleanup_known('a'*64))

    def test_deleted_leaf_needs_pinned_ancestor_wait_pidfd_and_headroom(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder);leaf=parent/'docker-owned.scope';leaf.mkdir()
            observer=object.__new__(AncestorObserver)
            observer.parent=parent;observer.leaf=leaf;observer.leaf_identity=identity(leaf)
            observer.poller=Mock();observer.poller.poll.return_value=[(12,1)]
            observer.peers=set();observer.pin={'identity':identity(parent),'oom_kill':0,
                'memory_max':'max','host_floor_bytes':44*1024**3,'ancestor_floor_bytes':44*1024**3}
            (parent/'memory.events').write_text('oom_kill 0\n')
            (parent/'memory.current').write_text(str(30*1024**3))
            (parent/'memory.max').write_text('max')
            leaf.rmdir()
            state={'Running':False,'Pid':0,'Restarting':False,'Dead':False,
                   'FinishedAt':'2026-10-08T11:00:00Z','ExitCode':137}
            original=Path.read_text
            def read(path,*args,**kwargs):
                if str(path)=='/proc/meminfo':return 'MemAvailable: 104857600 kB\n'
                return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',read):
                self.assertTrue(observer.sample(state,137).valid())
                observer.poller.poll.return_value=[]
                self.assertFalse(observer.sample(state,137).valid())
                observer.poller.poll.return_value=[(12,1)]
                with self.assertRaises(ValueError):observer.sample(state,0)
                (parent/'unexplained.scope').mkdir()
                self.assertFalse(observer.sample(state,137).valid())
                (parent/'unexplained.scope').rmdir()
                (parent/'memory.events').write_text('oom_kill 1\n')
                self.assertFalse(observer.sample(state,137).valid())
                (parent/'memory.events').unlink()
                with self.assertRaises(OSError):observer.sample(state,137)

    def test_wrong_identity_cannot_signal(self):
        stage=self.bare_stage()
        with patch.object(stage,'rpc') as rpc,self.assertRaises(ValueError):stage.terminate('b'*64,'KILL',1)
        rpc.assert_not_called()


if __name__=='__main__':unittest.main()
