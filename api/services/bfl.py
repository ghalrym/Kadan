"""BFL asynchronous submission and polling without automatic paid retries."""
import os
import time
from urllib.parse import urlsplit

import httpx

from api.inference.resources import ResourceCancelled


class BflFailure(RuntimeError):
    pass


def polling_url(value):
    """Accept only HTTPS result endpoints belonging to the documented BFL API."""
    parsed = urlsplit(value)
    host = parsed.hostname or ''
    if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not (host == 'api.bfl.ai' or (host.startswith('api.') and host.endswith('.bfl.ai')))
            or parsed.path != '/v1/get_result' or parsed.fragment):
        raise BflFailure('BFL returned an invalid polling URL.')
    return value


class BflClient:
    def __init__(self, client=None, timeout=1800):
        self.client = client
        self.timeout = timeout

    @staticmethod
    def configured():
        return bool(os.environ.get('BFL_API_KEY', '').strip())

    def generate(self, endpoint, payload, cancel):
        """Submit once and poll cooperatively; cancellation cannot revoke upstream billing."""
        key = os.environ.get('BFL_API_KEY', '').strip()
        if not key:
            raise BflFailure('BFL_API_KEY is not configured.')
        if endpoint not in ('flux-3-image', 'flux-3-video'):
            raise ValueError('Unsupported BFL endpoint')
        if cancel.is_set():
            raise ResourceCancelled('Generation cancelled before submission')
        if self.client is not None:
            return self._generate(self.client, key, endpoint, payload, cancel)
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            return self._generate(client, key, endpoint, payload, cancel)

    def _generate(self, client, key, endpoint, payload, cancel):
        try:
            response = client.post(f'https://api.bfl.ai/v1/{endpoint}',
                                   headers={'x-key': key}, json=payload)
            response.raise_for_status()
            submitted = response.json()
            url = polling_url(submitted['polling_url'])
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if cancel.is_set():
                    raise ResourceCancelled('Local wait cancelled; the submitted BFL job may still run and incur cost.')
                response = client.get(url, headers={'x-key': key})
                response.raise_for_status()
                result = response.json()
                status = result.get('status')
                if status == 'Ready':
                    sample = result.get('result', {}).get('sample')
                    parsed = urlsplit(sample) if isinstance(sample, str) else None
                    if parsed is None or parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
                        raise BflFailure('BFL returned an invalid result URL.')
                    return sample
                if status not in ('Pending', 'Reasoning', 'Generating'):
                    raise BflFailure(f'BFL generation ended with status {status!r}.')
                cancel.wait(1)
            raise BflFailure('BFL polling timed out; the submitted job may still run and incur cost.')
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            # Never include response bodies, API keys or signed result URLs.
            raise BflFailure('BFL request failed; a submitted job may still run. No automatic resubmission was made.') from None

    def download(self, url, path, cancel, max_bytes):
        """Stream signed output without forwarding the API credential or following redirects."""
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            self._download(self.client or client, url, path, cancel, max_bytes)

    @staticmethod
    def _download(client, url, path, cancel, max_bytes):
        try:
            with client.stream('GET', url) as response:
                response.raise_for_status()
                total = 0
                with path.open('wb') as output:
                    for chunk in response.iter_bytes():
                        if cancel.is_set():
                            raise ResourceCancelled('Result download cancelled')
                        total += len(chunk)
                        if total > max_bytes:
                            raise BflFailure('BFL result exceeds the output storage limit.')
                        output.write(chunk)
            if total == 0:
                raise BflFailure('BFL returned an empty result.')
        except httpx.HTTPError:
            path.unlink(missing_ok=True)
            raise BflFailure('BFL output download failed.') from None
        except BaseException:
            path.unlink(missing_ok=True)
            raise
