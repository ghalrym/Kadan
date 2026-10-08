"""Fail-closed outer stage supervisor. Transport is deliberately injected.

This module never creates containers, stops services, or restores production.
A reviewed window driver must implement OwnedStage and consume the verdict.
Successful TERM/KILL RPCs are not evidence that the owned container exited.
"""
from dataclasses import dataclass, asdict
import json
import re
import time
from typing import Protocol

from .artifacts import ExclusiveOutput, require


@dataclass(frozen=True)
class Observation:
    container_id: str
    running: bool
    process_count: int
    compute_process_count: int
    child_reaped: bool
    gpu_baseline_restored: bool
    memory_released: bool
    docker_oom: bool
    cgroup_oom_kill_delta: int
    manifest_unchanged: bool
    memory_current_bytes: int | None = None
    cgroup_populated: int | None = None

    def cleanup_known(self, owned_id):
        # Strict types prevent None, unknown counts or strings from clearing ownership.
        return (self.container_id == owned_id and self.running is False
                and type(self.process_count) is int and self.process_count == 0
                and type(self.compute_process_count) is int and self.compute_process_count == 0
                and self.child_reaped is True and self.gpu_baseline_restored is True
                and self.memory_released is True
                and type(self.memory_current_bytes) is int and self.memory_current_bytes == 0
                and type(self.cgroup_populated) is int and self.cgroup_populated == 0
                and type(self.docker_oom) is bool
                and type(self.cgroup_oom_kill_delta) is int and self.cgroup_oom_kill_delta >= 0
                and type(self.manifest_unchanged) is bool)


class OwnedStage(Protocol):
    """Every method is bounded by timeout; errors mean unknown, never success.

    The concrete driver must authenticate the immutable owned ID/image/labels,
    cgroup/resource limits, mounts and child command before start. It must retain
    raw stdout/stderr under bounded exclusive files, reap its client subprocesses,
    and read both Docker state and cgroup memory.events. observe must establish
    actual exit and absence of owned processes, memory/compute residency against
    the stopped baseline; it cannot infer these from an RPC return code.
    """
    def verify(self, container_id: str, timeout: float): ...
    def start(self, container_id: str, timeout: float): ...
    def poll(self, container_id: str, timeout: float): ...  # None or actual exec exit code
    def terminate(self, container_id: str, signal: str, timeout: float): ...
    def observe(self, container_id: str, timeout: float) -> Observation: ...
    def validate_artifacts(self, container_id: str, timeout: float): ...  # raises on missing DONE/hash/diagnostic


@dataclass(frozen=True)
class Policy:
    deadline: int = 1815
    term_grace: int = 5
    cleanup_deadline: int = 30
    stable_samples: int = 5
    command_timeout: int = 5

    def __post_init__(self):
        for value, ceiling in ((self.deadline,1815), (self.term_grace,5),
                               (self.cleanup_deadline,30), (self.command_timeout,5)):
            require(type(value) is int and 1 <= value <= ceiling, 'supervisor_deadline')
        require(self.stable_samples == 5, 'cleanup_samples')


def supervise(stage: OwnedStage, container_id, result_path, *, execute=False,
              policy=Policy(), clock=time.monotonic, sleep=time.sleep, cancelled=lambda: False):
    require(execute is True, 'supervisor_execution_intent')
    require(re.fullmatch('[0-9a-f]{64}', container_id) is not None, 'immutable_container_id')
    result = ExclusiveOutput(result_path)
    events = []
    report = {'container_id':container_id, 'accepted':False, 'cleanup_verified':False,
              'restoration_permitted':False, 'exit_code':None, 'events':events}
    owned = False
    errors = []
    def call(name, *args, budget=None):
        timeout = policy.command_timeout if budget is None else min(policy.command_timeout, budget)
        require(timeout > 0, 'supervisor_budget_expired')
        return getattr(stage,name)(container_id,*args,timeout=timeout)
    def fail(label, error):
        errors.append(label + ':' + str(error)[:1024])
    def cleanup_sleep(seconds):
        # Injected clock/sleep interruptions must not skip escalation or evidence.
        try:
            sleep(seconds)
        except BaseException as error:
            fail('cleanup_interrupted',error)

    try:
        try:
            require(not cancelled(), 'operator_cancelled')
            require(call('verify') is True, 'ownership_verification')
            # Ownership is retained even when start RPC fails after creating a child.
            owned = True
            end = clock() + policy.deadline
            require(not cancelled(), 'operator_cancelled')
            call('start', budget=end-clock())
            while clock() < end:
                require(not cancelled(), 'operator_cancelled')
                code = call('poll', budget=end-clock())
                if code is not None:
                    require(type(code) is int, 'invalid_exit_status')
                    report['exit_code'] = code
                    break
                sleep(min(1, max(0, end-clock())))
            if report['exit_code'] is None:
                errors.append('stage_deadline_or_unknown_exit')
            elif report['exit_code'] != 0:
                errors.append('stage_nonzero_exit')
            else:
                require(call('validate_artifacts') is True, 'artifact_validation')
        except BaseException as error:
            fail('stage',error)
        if owned:
            # Even normal exec exit leaves the container's sleep PID1 alive.
            # TERM/KILL target only this authenticated ID, never service identities.
            try:
                call('terminate','TERM')
                events.append('TERM_requested')
            except BaseException as error:
                fail('TERM',error)
            grace_end = clock() + policy.term_grace
            stopped = False
            while clock() < grace_end:
                try:
                    stopped = call('observe',budget=grace_end-clock()).cleanup_known(container_id)
                except BaseException as error:
                    fail('TERM_observe',error)
                if stopped:
                    break
                cleanup_sleep(min(1,max(0,grace_end-clock())))
            if not stopped:
                try:
                    call('terminate','KILL')
                    events.append('KILL_requested_not_proof_of_exit')
                except BaseException as error:
                    fail('KILL',error)
            end = clock() + policy.cleanup_deadline
            stable = 0
            while clock() < end:
                try:
                    observation = call('observe',budget=end-clock())
                    report['last_observation'] = asdict(observation)
                    if observation.docker_oom is not False or observation.cgroup_oom_kill_delta != 0:
                        if 'OOM_or_unknown' not in errors:
                            errors.append('OOM_or_unknown')
                    if observation.manifest_unchanged is not True:
                        if 'manifest_changed_or_unknown' not in errors:
                            errors.append('manifest_changed_or_unknown')
                    stable = stable + 1 if observation.cleanup_known(container_id) else 0
                    if stable == policy.stable_samples:
                        report['cleanup_verified'] = True
                        break
                except BaseException as error:
                    stable = 0
                    fail('cleanup_observe',error)
                cleanup_sleep(min(1,max(0,end-clock())))
            if not report['cleanup_verified']:
                errors.append('cleanup_uncertain_hold_ownership_no_restoration')
        if cancelled():
            errors.append('operator_cancelled')
        # This is only the cleanup prerequisite, never service-window authorization.
        report['restoration_permitted'] = report['cleanup_verified']
        report['accepted'] = report['cleanup_verified'] and not errors and report['exit_code'] == 0
        report['errors'] = errors
        data = json.dumps(report,allow_nan=False,indent=2).encode() + b'\n'
        require(len(data) <= 128 * 1024, 'supervisor_report_bound')
        result.write(data)
        result.finish()
        return report
    finally:
        result.close()


def main():
    # Delayed transport import keeps the state machine usable without process I/O.
    import argparse
    import signal
    from threading import Event
    from .container_stage import DockerReferenceStage
    parser=argparse.ArgumentParser(description='Supervise an already-created owned CPU reference container; never restore a service.')
    parser.add_argument('--execute-approved-reference',action='store_true')
    for name in ('container-id','run-id','manifest','manifest-sha256','source','snapshot','evidence'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:
        require(args.execute_approved_reference,'supervisor_execution_intent')
        cancellation=Event()
        def mark_cancelled(signum, frame):
            cancellation.set()
        signal.signal(signal.SIGTERM,mark_cancelled)
        signal.signal(signal.SIGINT,mark_cancelled)
        stage=DockerReferenceStage(args.manifest,args.manifest_sha256,args.run_id,
                                   args.source,args.snapshot,args.evidence)
        report=supervise(stage,args.container_id,stage.evidence/'supervisor.json',execute=True,
                         cancelled=cancellation.is_set)
        print(json.dumps(report))
        return 0 if report['accepted'] else (1 if report['cleanup_verified'] else 2)
    except BaseException as error:
        print(json.dumps({'accepted':False,'restoration_permitted':False,'error':str(error)[:4096]}))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
