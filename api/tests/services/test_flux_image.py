from io import BytesIO
import threading
import unittest
from unittest.mock import Mock

from PIL import Image

from api.services.flux_image import FluxImageProvider


class FluxImageTests(unittest.TestCase):
    def test_actual_payload_decode_and_no_local_inference(self):
        client = Mock()
        client.configured.return_value = True
        client.generate.return_value = 'https://media.example.test/image'
        def download(url, path, cancel, limit):
            Image.new('RGB', (2, 2), 'red').save(path, format='JPEG')
        client.download.side_effect = download
        provider = FluxImageProvider(client)
        images = provider.generate('a red square', '1:1', 2, threading.Event(), source=Image.new('RGB', (2, 2)))
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].size, (2, 2))
        endpoint, payload, cancel = client.generate.call_args.args
        self.assertEqual(endpoint, 'flux-3-image')
        self.assertEqual(payload['resolution'], '1k')
        self.assertTrue(payload['images'][0].startswith('data:image/png;base64,'))
        self.assertNotIn('seed', payload)

    def test_seed_rejected_before_paid_request(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, 'seed'):
            FluxImageProvider(client).generate('test', '1:1', 1, threading.Event(), seed=42)
        client.generate.assert_not_called()

    def test_missing_configuration_and_invalid_controls(self):
        client = Mock()
        client.configured.return_value = False
        with self.assertRaises(RuntimeError):
            FluxImageProvider(client).generate('test', '1:1', 1, threading.Event())
        for aspect, count in [('bad', 1), ('1:1', 0), ('1:1', 5)]:
            with self.assertRaises(ValueError):
                FluxImageProvider(client).generate('test', aspect, count, threading.Event())
        client.generate.assert_not_called()
