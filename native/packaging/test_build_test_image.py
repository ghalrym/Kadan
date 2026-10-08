"""Stdlib checks: no Docker, GPU, checkpoint or service calls."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('build_test_image', Path(__file__).with_name('build_test_image.py'))
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


class PackagingTests(unittest.TestCase):
    toolchain = 'registry.example/cuda-toolchain@sha256:' + 'a' * 64
    api = 'registry.example/kadan-api@sha256:' + 'b' * 64
    revision = 'c' * 40

    def test_exact_commands(self):
        archive, build = packaging.commands(self.toolchain, self.api, self.revision, 'kadan-native:test')
        self.assertEqual(archive, ['git', 'archive', '--format=tar', self.revision, 'api', 'native'])
        self.assertEqual(build, ['docker', 'build', '--pull=false', '--platform=linux/amd64', '--file', 'native/packaging/Dockerfile',
                               '--build-arg', 'TOOLCHAIN_IMAGE=' + self.toolchain,
                               '--build-arg', 'API_IMAGE=' + self.api,
                               '--build-arg', 'SOURCE_REVISION=' + self.revision,
                               '--tag', 'kadan-native:test', '-'])

    def test_unpinned_or_injected_inputs_rejected(self):
        for image in ['cuda:latest', 'sha256:' + 'a' * 64, self.api + '\n', self.api + ';id', '-bad@sha256:' + 'a' * 64]:
            with self.subTest(image=image), self.assertRaises(ValueError):
                packaging.commands(image, self.api, self.revision, 'test:image')
        for revision in ['HEAD', 'main', 'c' * 39, '--help', self.revision + '\n']:
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                packaging.commands(self.toolchain, self.api, revision, 'test:image')
        for tag in ['--push', 'bad tag', 'test;id', 'test\n']:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                packaging.commands(self.toolchain, self.api, self.revision, tag)

    def test_default_is_inert(self):
        output = io.StringIO()
        with patch.object(packaging.subprocess, 'run', side_effect=AssertionError('external action')), \
             patch.object(packaging.subprocess, 'check_output', side_effect=AssertionError('external action')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(packaging.main(['--toolchain-image', self.toolchain, '--api-image', self.api,
                                             '--revision', self.revision, '--tag', 'native:test']), 0)
        self.assertFalse(json.loads(output.getvalue())['executes_build'])

    def test_build_uses_immutable_archive_and_no_shell(self):
        with patch.object(packaging.subprocess, 'check_output', return_value=self.revision + '\n'), \
             patch.object(packaging.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            packaging.main(['--toolchain-image', self.toolchain, '--api-image', self.api,
                            '--revision', self.revision, '--tag', 'native:test', '--build'])
        calls = run.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].args[0][:3], ['git', 'cat-file', '-e'])
        self.assertEqual(calls[1].args[0][:3], ['git', 'archive', '--format=tar'])
        self.assertEqual(calls[2].args[0][:2], ['docker', 'build'])
        self.assertIs(calls[1].kwargs['stdout'], calls[2].kwargs['stdin'])
        for call in calls:
            self.assertTrue(call.kwargs['check'])
            self.assertNotIn('shell', call.kwargs)

    def test_archive_failure_prevents_docker(self):
        failure = packaging.subprocess.CalledProcessError(1, ['git', 'archive'])
        with patch.object(packaging.subprocess, 'check_output', return_value=self.revision + '\n'), \
             patch.object(packaging.subprocess, 'run', side_effect=[None, failure]) as run, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(packaging.subprocess.CalledProcessError):
            packaging.main(['--toolchain-image', self.toolchain, '--api-image', self.api,
                            '--revision', self.revision, '--tag', 'native:test', '--build'])
        self.assertEqual(run.call_count, 2)

    def test_default_local_image_has_explicit_verified_tag(self):
        output = io.StringIO()
        with patch.object(packaging.subprocess, 'run', side_effect=AssertionError('external action')), \
             patch.object(packaging.subprocess, 'check_output', side_effect=AssertionError('external action')), \
             contextlib.redirect_stdout(output):
            packaging.main(['--revision', self.revision, '--tag', 'native:test'])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['prepare_local_base'][0][3], packaging.DEFAULT_API)
        self.assertEqual(plan['prepare_local_base'][1][:3], ['docker', 'image', 'tag'])
        self.assertIn('API_IMAGE=kadan-native-api-base:' + packaging.DEFAULT_API[7:], plan['build'])

    def test_wrong_local_image_identity_stops_before_tag_or_build(self):
        with patch.object(packaging.subprocess, 'check_output', side_effect=[self.revision, 'sha256:' + 'd' * 64]), \
             patch.object(packaging.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'identity mismatch'):
            packaging.main(['--revision', self.revision, '--tag', 'native:test', '--build'])
        self.assertEqual(run.call_count, 1)  # git cat-file only

    def test_local_build_verifies_image_before_tag_and_after_build(self):
        with patch.object(packaging.subprocess, 'check_output', side_effect=[self.revision, packaging.DEFAULT_API, packaging.DEFAULT_API]) as inspect, \
             patch.object(packaging.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            packaging.main(['--revision', self.revision, '--tag', 'native:test', '--build'])
        self.assertEqual(run.call_args_list[1].args[0][:3], ['docker', 'image', 'tag'])
        self.assertEqual(run.call_args_list[-1].args[0][:2], ['docker', 'build'])
        self.assertEqual(inspect.call_count, 3)
        self.assertEqual(inspect.call_args_list[-1].args[0][3], 'kadan-native-api-base:' + packaging.DEFAULT_API[7:])

    def test_retagged_base_after_build_is_failure(self):
        with patch.object(packaging.subprocess, 'check_output', side_effect=[self.revision, packaging.DEFAULT_API, 'sha256:' + 'd' * 64]), \
             patch.object(packaging.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(ValueError, 'discard output'):
            packaging.main(['--revision', self.revision, '--tag', 'native:test', '--build'])


if __name__ == '__main__':
    unittest.main()
