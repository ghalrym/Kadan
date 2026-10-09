"""Execute the real harness wiring without importing its GPU entry point."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from api.services.image_jobs import ImageJobs


class ImageHarnessTests(unittest.TestCase):
    def test_live_switch_constructs_and_uses_instrumented_generator(self):
        path = Path(__file__).resolve().parents[3] / 'docs/benchmarks/image-compile-119/live_switch.py'
        tree = ast.parse(path.read_text())
        run = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'run')
        # Run the harness's actual constructor/injection statements, not copies.
        wiring = [node for node in run.body if isinstance(node, ast.Assign) and
                  any((isinstance(target, ast.Name) and target.id == 'IMAGE') or
                      (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and
                       target.value.id == 'IMAGE') for target in node.targets)]
        self.assertEqual(len(wiring), 2)
        generator = Mock()
        factory = Mock(return_value=generator)
        runtime, downloads = object(), object()
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(ImageJobs=ImageJobs, ROOT=Path(directory), model_manager=downloads,
                RUNTIME=runtime, QwenImagePipeline=factory, image_path=Path(directory),
                RESOURCES=object(), MANIFEST={'image_offload_mode': 'sequential'})
            exec(compile(ast.Module(body=wiring, type_ignores=[]), str(path), 'exec'), namespace)
            jobs = namespace['IMAGE']
            self.assertIs(jobs.chat_runtime, runtime)
            self.assertIs(jobs.downloads, downloads)
            self.assertIs(jobs.generator, generator)
            jobs.offload_to_ram()
            generator.offload_to_ram.assert_called_once_with(None)
            jobs.close()
            generator.close.assert_called_once_with()
        # Profiling and post-handoff assertions must use the same owned object.
        attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                      and isinstance(node.value, ast.Name) and node.value.id == 'IMAGE']
        self.assertNotIn('native', attributes)
        self.assertGreaterEqual(attributes.count('generator'), 5)
