import io
import json
from pathlib import Path
import runpy
import unittest
from contextlib import redirect_stdout

from api.server import app


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "frontend" / "openapi.json"
REGENERATE = ".venv/bin/python -m api.export_openapi > frontend/openapi.json"


class OpenApiExportTests(unittest.TestCase):
    def test_schema_file_matches_app(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema,
            app.openapi(),
            f"OpenAPI schema is stale. Run from the repository root: {REGENERATE}",
        )

    def test_export_matches_schema_file(self):
        output = io.StringIO()
        with redirect_stdout(output):
            runpy.run_module("api.export_openapi", run_name="__main__")

        self.assertEqual(
            output.getvalue(),
            SCHEMA_PATH.read_text(encoding="utf-8"),
            f"Committed schema differs from generated output. Run: {REGENERATE}",
        )
