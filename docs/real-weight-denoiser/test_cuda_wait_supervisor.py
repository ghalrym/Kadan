import ast
import json
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import launch_cuda_wait as launch
from supervisor_cuda_wait import cleanup


class WaitSupervisorTests(unittest.TestCase):
    def test_cleanup_signals_both_before_waiting_under_one_deadline(self):
        events=[]
        children=[Mock(pid=100),Mock(pid=101)]
        for child in children:child.wait.side_effect=lambda timeout:events.append(('wait',timeout))
        with patch('supervisor_cuda_wait.os.killpg',side_effect=lambda pid,sig:events.append(('kill',pid))), \
             patch('supervisor_cuda_wait.time.monotonic',side_effect=[11,12]):
            cleanup(children,20)
        self.assertEqual(events,[('kill',100),('kill',101),('wait',9),('wait',8)])

    def test_running_loop_enforces_resource_admission(self):
        # Execute the actual running-container loop only, with synthetic process
        # telemetry; no preflight, Docker launch or GPU work is evaluated.
        tree=ast.parse(Path(launch.__file__).read_text())
        loop=next(node for node in ast.walk(tree) if isinstance(node,ast.While)
            and 'inspect_container' in ast.unparse(node.test))
        guard=Mock(side_effect=RuntimeError('Host free memory guard'))
        namespace=dict(host=SimpleNamespace(inspect_container=lambda name:{'State':{'Running':True}},guards=guard),
            NAME='fixture',time=SimpleNamespace(monotonic=lambda:0),stage_end=1,measurement_end=1)
        with self.assertRaisesRegex(RuntimeError,'Host free memory guard'):
            exec(compile(ast.Module(body=[loop],type_ignores=[]),'running-loop','exec'),namespace)
        guard.assert_called_once()

    def test_review_binds_exact_settings_and_head(self):
        record=dict(protocol=launch.PROTOCOL,source_commit='a'*40,case='control',
                    settings=launch.settings('control'),criteria=launch.CRITERIA,
                    decision='approved-for-bounded-execution',reviewer='reviewer',review_reference='reference')
        launch.require_review(record,'a'*40,'control')
        for key in ('source_commit','settings','criteria','decision','reviewer'):
            with self.assertRaises(ValueError):launch.require_review(dict(record,**{key:None}),'a'*40,'control')

    def test_output_requires_physical_identity(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            for device in (0,1):
                (path/f'rank-{device}.json').write_text(json.dumps(dict(device=device,policy='control',
                    commit='a'*40,gpu_uuid=launch.host.GPUS[device],flags={'before':0,'after':0,'after_window':0})))
            launch.verify(path,'control','a'*40)
            row=json.loads((path/'rank-1.json').read_text());row['gpu_uuid']=launch.host.GPUS[0]
            (path/'rank-1.json').write_text(json.dumps(row))
            with self.assertRaises(ValueError):launch.verify(path,'control','a'*40)


if __name__ == '__main__': unittest.main()
