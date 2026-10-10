import asyncio
import unittest
from api.inference.progress import track, report, snapshot


class ProgressTests(unittest.TestCase):
    def test_thread_context_and_cleanup(self):
        async def run():
            with track('job', 'image'):
                await asyncio.to_thread(report, 'text_loading_mib', 64)
                value = snapshot()
                self.assertEqual(value['stage'], 'text_loading_mib')
                self.assertEqual(value['value'], 64)
                value['stage'] = 'changed'
                self.assertEqual(snapshot()['stage'], 'text_loading_mib')
            self.assertIsNone(snapshot())
        asyncio.run(run())

    def test_failure_clears_active_progress(self):
        with self.assertRaises(RuntimeError):
            with track('job', 'image'):
                report('preparing')
                raise RuntimeError('failed')
        self.assertIsNone(snapshot())
