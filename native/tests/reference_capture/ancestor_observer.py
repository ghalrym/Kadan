"""Read-only existing-ancestor termination/headroom evidence for a deleted Docker leaf.

No zero-freed-byte claim, new cgroup, reclaim, or host setting mutation.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import select

from .artifacts import require


@dataclass(frozen=True)
class HeadroomProof:
    init_terminated: bool
    absence_verified: bool
    ancestor_identity_unchanged: bool
    ancestor_oom_unchanged: bool
    ancestor_peers_unchanged: bool
    wait_exit_code: int
    host_available_bytes: int
    host_floor_bytes: int
    ancestor_current_bytes: int
    ancestor_limit_bytes: int | None
    ancestor_floor_bytes: int

    def valid(self):
        flags=(self.init_terminated,self.absence_verified,self.ancestor_identity_unchanged,
               self.ancestor_oom_unchanged,self.ancestor_peers_unchanged)
        numbers=(self.host_available_bytes,self.host_floor_bytes,self.ancestor_current_bytes,self.ancestor_floor_bytes)
        return (all(v is True for v in flags) and type(self.wait_exit_code) is int
                and all(type(v) is int and v>=0 for v in numbers)
                and self.host_floor_bytes>=44*1024**3 and self.ancestor_floor_bytes>=44*1024**3
                and self.host_available_bytes>=self.host_floor_bytes
                and (self.ancestor_limit_bytes is None or
                     (type(self.ancestor_limit_bytes) is int and
                      self.ancestor_limit_bytes-self.ancestor_current_bytes>=self.ancestor_floor_bytes)))


def counter(path, key):
    fields=dict(line.split() for line in path.read_text().splitlines())
    value=fields[key]
    require(value.isdecimal(), 'observer_counter')
    return int(value)


def identity(path):
    st=path.stat()
    require(not path.is_symlink(), 'observer_symlink')
    return [st.st_dev,st.st_ino]


def require_hierarchical_events(mountinfo=None):
    if mountinfo is None:
        mountinfo=Path('/proc/self/mountinfo').read_text()
    matches=[]
    for line in mountinfo.splitlines():
        fields=line.split()
        if ' - ' not in line:
            continue
        separator=fields.index('-')
        if len(fields)>separator+3 and fields[4]=='/sys/fs/cgroup' and fields[separator+1]=='cgroup2':
            matches.append(set(fields[5].split(',')) | set(fields[separator+3].split(',')))
    require(len(matches)==1, 'cgroup2_mount_evidence')
    require('memory_localevents' not in matches[0], 'nonhierarchical_memory_events')


class AncestorObserver:
    def __init__(self, leaf, pid, pin):
        require_hierarchical_events()
        self.leaf=leaf
        self.parent=leaf.parent
        require(str(self.parent)==pin['path']=='/sys/fs/cgroup/system.slice','existing_ancestor_only')
        require(identity(self.parent)==pin['identity'],'ancestor_identity')
        require(counter(self.parent/'memory.events','oom_kill')==pin['oom_kill'],'ancestor_oom_baseline')
        require(type(pin['memory_current_ceiling']) is int and
                int((self.parent/'memory.current').read_text())<=pin['memory_current_ceiling'], 'ancestor_quiet_baseline')
        available=Path('/proc/meminfo').read_text().split('MemAvailable:',1)[1].splitlines()[0].split()
        require(len(available)==2 and available[1]=='kB' and int(available[0])*1024>=pin['host_floor_bytes']>=44*1024**3,'admission_host_headroom')
        limit=(self.parent/'memory.max').read_text().strip()
        require(limit==pin['memory_max'] and pin['ancestor_floor_bytes']>=44*1024**3,'admission_ancestor_limit')
        require(limit=='max' or int(limit)-int((self.parent/'memory.current').read_text())>=pin['ancestor_floor_bytes'],'admission_ancestor_headroom')
        self.pin=pin
        self.leaf_identity=identity(leaf)
        self.peers={p.name for p in self.parent.iterdir() if p.is_dir() and p!=leaf}
        require(sorted(self.peers)==pin['peers'],'ancestor_peers_baseline')
        self.pidfd=os.pidfd_open(pid,0)  # Unsupported or denied => fail before launch.
        self.poller=select.poll()
        self.poller.register(self.pidfd,select.POLLIN)
        require(not self.poller.poll(0),'init_already_exited')

    def sample(self, state, wait_exit_code):
        require_hierarchical_events()
        require(state['Running'] is False and state['Pid']==0 and state['Restarting'] is False
                and state['Dead'] is False and state['FinishedAt']!='0001-01-01T00:00:00Z', 'stopped_metadata')
        require(type(wait_exit_code) is int and wait_exit_code==state['ExitCode'], 'docker_wait_exit')
        require(identity(self.parent)==self.pin['identity'],'ancestor_identity')
        terminated=bool(self.poller.poll(0))
        absent=False
        try:
            current=identity(self.leaf)
        except FileNotFoundError:
            # Only ENOENT with the pinned parent intact proves deletion. Other
            # errors propagate; a replaced leaf is never the original workload.
            require(identity(self.parent)==self.pin['identity'],'ancestor_identity')
            absent=True
        else:
            require(current==self.leaf_identity,'leaf_replaced')
            absent=((self.leaf/'cgroup.procs').read_text().strip()=='' and
                    counter(self.leaf/'cgroup.events','populated')==0)
        peers={p.name for p in self.parent.iterdir() if p.is_dir() and p!=self.leaf}
        oom=counter(self.parent/'memory.events','oom_kill')
        current=int((self.parent/'memory.current').read_text())
        raw_limit=(self.parent/'memory.max').read_text().strip()
        require(raw_limit==self.pin['memory_max'],'ancestor_limit_changed')
        limit=None if raw_limit=='max' else int(raw_limit)
        meminfo=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        value,unit=meminfo['MemAvailable'].split()
        require(unit=='kB','memavailable_units')
        return HeadroomProof(terminated,absent,identity(self.parent)==self.pin['identity'],
                             oom==self.pin['oom_kill'],peers==self.peers,wait_exit_code,
                             int(value)*1024,self.pin['host_floor_bytes'],current,limit,
                             self.pin['ancestor_floor_bytes'])

    def close(self):
        os.close(self.pidfd)
