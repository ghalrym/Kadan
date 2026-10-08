"""CPU-reference Docker transport for a pre-created, explicitly owned container.

No create/start-service/restart/restore operation exists here. A stopped cgroup
that disappears before a final memory.events sample leaves cleanup uncertain.
"""
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import tempfile
import sys
import time

from .artifacts import ExclusiveOutput, require, sha256
from .capture_actual import bounded_json, validate_manifest, plain_path, BUFFER
from .capture import REPO, PINS
from .supervisor import Observation

MAX_LOG = 8 * BUFFER


def log_limit():
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_LOG, MAX_LOG))


class DockerReferenceStage:
    def __init__(self, manifest, manifest_digest, run_id, source, snapshot, evidence):
        self.manifest_path = plain_path(manifest)
        self.manifest_digest = manifest_digest
        self.manifest = bounded_json(self.manifest_path, BUFFER, manifest_digest)
        validate_manifest(self.manifest)
        require(run_id and len(run_id) <= 128, 'run_id')
        self.run_id = run_id
        self.source, self.snapshot, self.evidence = map(plain_path,(source,snapshot,evidence))
        require(self.source == REPO, 'reviewed_source_mount')
        require(self.snapshot.name == self.manifest['snapshot'], 'snapshot_mount')
        require(self.manifest_path.parent == self.evidence, 'manifest_location')
        require(self.evidence.stat().st_mode & 0o777 == 0o700, 'evidence_mode')
        self.child = None
        self.logs = []
        self.evidence_children = []
        self.cgroup = None
        self.initial_oom = None
        self.final_oom = None
        self.cgroup_identity = None
        self.observer_parent = False
        self.owned_cgroup_name = None
        self.verified_id = None

    def rpc(self, args, timeout):
        # Bounded disk-backed capture: no unbounded communicate() buffers.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            completed = subprocess.run(['docker',*args],stdout=out,stderr=err,
                                       timeout=timeout,check=False,preexec_fn=log_limit)
            require(completed.returncode == 0, 'docker_rpc:' + args[0])
            require(out.tell() <= MAX_LOG and err.tell() <= MAX_LOG, 'docker_rpc_output')
            out.seek(0)
            return out.read(MAX_LOG+1)

    def inspect(self, cid, timeout):
        values=json.loads(self.rpc(['inspect',cid],timeout))
        require(len(values)==1 and values[0]['Id']==cid, 'container_identity')
        return values[0]

    def oom(self):
        fields=dict(line.split() for line in (self.cgroup/'memory.events').read_text().splitlines())
        return int(fields['oom_kill'])

    def verify(self, cid, timeout):
        head=subprocess.run(['git','-C',str(self.source),'rev-parse','HEAD'],
                            capture_output=True,text=True,check=True,timeout=timeout).stdout.strip()
        require(head==self.manifest['source_revision'], 'source_head')
        validate_manifest(self.manifest)
        for name,digest in json.loads(PINS.read_text())['reference_sources'].items():
            require(sha256((self.source/name).read_bytes())==digest, 'reference_source')
        value=self.inspect(cid,timeout)
        host,config=value['HostConfig'],value['Config']
        require(value['Image']==self.manifest['image_id'], 'container_image')
        require(config.get('Labels',{}).get('kadan.reference.run')==self.run_id, 'container_ownership_label')
        require(value['State']['Running'] is True and value['Path']=='/bin/sleep'
                and value['Args']==['infinity'], 'inert_container_required')
        require(host['RestartPolicy']['Name']=='no' and host['NetworkMode']=='none'
                and host['ReadonlyRootfs'] is True and host['Privileged'] is False
                and host['PidMode']=='' and host['IpcMode']=='private', 'container_isolation')
        require(host.get('Runtime')=='runc' and not host.get('DeviceRequests')
                and not host.get('Devices') and not host.get('DeviceCgroupRules'), 'container_devices')
        require(0 < host['Memory'] <= 40*1024**3 and host['MemorySwap']==host['Memory']
                and 0 < host['NanoCpus'] <= 2*10**9 and 0 < host['PidsLimit'] <= 256, 'container_limits')
        require(host.get('CapDrop')==['ALL'] and not host.get('CapAdd')
                and 'no-new-privileges' in host.get('SecurityOpt',[]), 'container_privileges')
        env=dict(item.split('=',1) for item in config['Env'])
        require(not any(env.get(k) for k in ('LD_PRELOAD','LD_LIBRARY_PATH','PYTHONHOME','PYTHONPATH')), 'container_import_env')
        require(env.get('CUDA_VISIBLE_DEVICES')=='' and env.get('NVIDIA_VISIBLE_DEVICES')=='void', 'container_gpu_env')
        mounts={(p['Source'],p['Destination'],p['RW'],p['Type']) for p in value['Mounts']}
        expected={(str(self.source),'/work',False,'bind'),
                  (str(self.snapshot),'/snapshot/'+self.manifest['snapshot'],False,'bind'),
                  (str(self.evidence),'/evidence',True,'bind')}
        require(mounts==expected, 'container_mounts')
        pid=value['State']['Pid']
        lines=Path(f'/proc/{pid}/cgroup').read_text().splitlines()
        require(len(lines)==1 and lines[0].startswith('0::/'), 'cgroup_v2_required')
        relative=lines[0][4:]
        require(cid in relative and '..' not in Path(relative).parts, 'owned_cgroup_path')
        self.cgroup=Path('/sys/fs/cgroup')/relative
        child_cgroup=self.cgroup
        parent_pin=self.manifest.get('retained_parent')
        if parent_pin is not None:
            name=parent_pin['name']
            require(re.fullmatch(r'kadanreference[0-9a-f]{16}\.slice',name) is not None, 'observer_parent_name')
            require(host.get('CgroupParent')==name, 'observer_parent_config')
            parent=Path('/sys/fs/cgroup')/name
            require(child_cgroup.parent==parent, 'observer_parent_membership')
            st=parent.stat()
            require([st.st_dev,st.st_ino]==parent_pin['identity'], 'observer_parent_identity')
            require((parent/'cgroup.procs').read_text().strip()=='', 'observer_parent_processes')
            self.cgroup=parent
            self.observer_parent=True
            self.owned_cgroup_name=child_cgroup.name
            self.verify_observer_children()
        st=self.cgroup.stat()
        self.cgroup_identity=(st.st_dev,st.st_ino)
        self.initial_oom=self.oom()
        require(self.initial_oom==0, 'initial_oom_history')
        procs=(child_cgroup/'cgroup.procs').read_text().split()
        require(procs==[str(pid)], 'unexpected_container_children')
        self.verified_id=cid
        return True

    def verify_observer_children(self):
        if not getattr(self,'observer_parent',False):
            return
        directories={p.name for p in self.cgroup.iterdir() if p.is_dir()}
        require(directories <= {self.owned_cgroup_name}, 'unowned_observer_descendant')
        for name in directories:
            require(not any(p.is_dir() for p in (self.cgroup/name).iterdir()), 'nested_observer_descendant')

    def start(self, cid, timeout):
        require(cid==self.verified_id and self.child is None, 'stage_start_once')
        try:
            for name in ('stdout.raw','stderr.raw'):
                self.logs.append(ExclusiveOutput(self.evidence/name))
            command=['docker','exec','-w','/work',cid,'python','-B','-m',
                     'native.tests.reference_capture.capture_actual','--execute-approved-actual-reference',
                     '--snapshot-root','/snapshot/'+self.manifest['snapshot'],
                     '--manifest','/evidence/'+self.manifest_path.name,
                     '--manifest-sha256',self.manifest_digest,'--output-dir','/evidence/capture']
            self.child=subprocess.Popen(command,stdout=self.logs[0].fd,stderr=self.logs[1].fd,
                                        stdin=subprocess.DEVNULL,preexec_fn=log_limit)
        except BaseException:
            for log in self.logs:log.close()
            raise

    def poll(self, cid, timeout):
        require(cid==self.verified_id and self.child is not None, 'stage_identity')
        code=self.child.poll()
        for log in self.logs:
            if log.fd>=0:require(os.fstat(log.fd).st_size < MAX_LOG, 'raw_output_limit')
        if code is not None:
            # Preserve streams even when the child failed; fsync errors fail the stage.
            for log in self.logs:
                if log.fd>=0:log.finish()
        return code

    def evidence_check(self, operation, path, timeout):
        # A regular-file read can still stall in the filesystem. Isolate it from
        # the control loop and retain ownership of helpers that cannot be reaped.
        end=time.monotonic()+timeout
        child=subprocess.Popen([sys.executable,'-B','-m',
                                'native.tests.reference_capture.evidence',operation,str(path),
                                self.manifest_digest],stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=REPO)
        self.evidence_children.append(child)
        try:
            code=child.wait(timeout=max(0,end-time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()
            # No unbounded wait/communicate after KILL: observation owns reaping.
            child.poll()
            raise
        require(code==0, 'evidence_validation:'+operation)
        return True

    def validate_artifacts(self, cid, timeout):
        require(cid==self.verified_id and self.child.poll()==0, 'child_exit')
        return self.evidence_check('artifacts',self.evidence/'capture',timeout)

    def terminate(self, cid, signal, timeout):
        require(cid==self.verified_id and signal in ('TERM','KILL'), 'owned_signal')
        # Capture events before teardown, but never substitute this sample for a final one.
        self.rpc(['kill','--signal',signal,cid],timeout)

    def observe(self, cid, timeout):
        require(cid==self.verified_id, 'owned_observation')
        end=time.monotonic()+timeout
        state=self.inspect(cid,timeout)['State']
        helpers_reaped=all(child.poll() is not None for child in self.evidence_children)
        reaped=(self.child is None or self.child.poll() is not None) and helpers_reaped
        if not state['Running'] and self.child is not None and self.child.poll() is None:
            # This terminates only our Docker client, not proof of container cleanup.
            self.child.kill()
            try:self.child.wait(timeout=max(0,end-time.monotonic()))
            except subprocess.TimeoutExpired:pass
            reaped=self.child.poll() is not None and helpers_reaped
        if reaped:
            for log in self.logs:
                if log.fd>=0:log.finish()
        # Require a final cgroup sample after the owned child has exited. On engines
        # that remove the cgroup immediately this intentionally returns uncertainty.
        procs=None
        memory_current=None
        populated=None
        self.final_oom=None  # Never reuse a previous sample after evidence vanishes.
        if self.cgroup.exists():
            self.verify_observer_children()
            before=self.cgroup.stat()
            require((before.st_dev,before.st_ino)==self.cgroup_identity, 'cgroup_replaced')
            procs=(self.cgroup/'cgroup.procs').read_text().split()
            # memory.current is hierarchical: resident descendants and retained
            # page-cache/kernel charges are included even when direct procs is empty.
            raw=(self.cgroup/'memory.current').read_text().strip()
            require(raw.isdecimal(), 'cgroup_memory_current')
            memory_current=int(raw)
            fields=dict(line.split() for line in (self.cgroup/'cgroup.events').read_text().splitlines())
            require(fields.get('populated') in ('0','1'), 'cgroup_population_unknown')
            populated=int(fields['populated'])  # Hierarchical, unlike cgroup.procs.
            if not procs and populated==0 and reaped:
                self.final_oom=self.oom()-self.initial_oom
            after=self.cgroup.stat()
            require((after.st_dev,after.st_ino)==self.cgroup_identity, 'cgroup_replaced')
        gone=not state['Running'] and state['Pid']==0 and procs==[] and populated==0 and reaped
        # Fixed conservative stopped baseline: exactly zero charged bytes, no
        # tolerance and no substitution of process absence for memory evidence.
        memory_released=gone and memory_current==0
        remaining=end-time.monotonic()
        require(remaining>0, 'observation_deadline')
        unchanged=self.evidence_check('manifest',self.manifest_path,remaining)
        return Observation(cid,state['Running'],len(procs) if procs is not None else None,
                           0,reaped,gone,memory_released,state['OOMKilled'],self.final_oom,unchanged,
                           memory_current,populated)
