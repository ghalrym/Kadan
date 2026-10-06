"""Run the pinned upstream filename function without importing its GPU runtime."""
import ast
import hashlib
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata
import unittest

from api.inference.h3 import sampling_arguments


@unittest.skipUnless(os.getenv('KADAN_H3_SAMPLING_SOURCE'), 'Pinned SGLang source supplied by CI')
class H3UpstreamTests(unittest.TestCase):
    def test_renderer_filename_survives_real_upstream_sanitizer(self):
        source = Path(os.environ['KADAN_H3_SAMPLING_SOURCE']).read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         'd55d620aefceac56d8179e1744462cbbbe18aee25218bce4c4ecc40747aabbae')
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_sanitize_filename')
        namespace = dict(re=re, unicodedata=unicodedata)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned-sglang-sanitizer>', 'exec'), namespace)
        sanitize = namespace['_sanitize_filename']
        self.assertEqual(sanitize('.job.partial.mp4'), 'job.partial.mp4')
        spec = SimpleNamespace(prompt='test', aspect='16:9', duration=4, seed=42, resolution='480p')
        arguments = sampling_arguments(spec, Path('/private/job/video.mp4'))
        self.assertEqual(sanitize(arguments['output_file_name']), arguments['output_file_name'])
