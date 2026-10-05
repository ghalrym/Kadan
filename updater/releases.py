"""Only paired digest releases from successful, exact-master Kadan workflows."""
import json
import re
from urllib.request import Request, urlopen

REPOSITORY = 'ghalrym/Kadan'
API = f'https://api.github.com/repos/{REPOSITORY}'
SHA = re.compile(r'[0-9a-f]{40}')
DIGEST = re.compile(r'sha256:[0-9a-f]{64}')


class ReleaseError(RuntimeError):
    pass


def fetch_json(url):
    request = Request(url, headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'Kadan-updater'})
    with urlopen(request, timeout=20) as response:
        data = response.read(256 * 1024 + 1)
    if len(data) > 256 * 1024:
        raise ReleaseError('Release response exceeds its size limit')
    return json.loads(data)


def validate_manifest(value):
    expected = {'format', 'commit', 'api', 'frontend', 'migrations', 'data_epoch', 'ci_run', 'publish_run'}
    if not isinstance(value, dict) or set(value) != expected or value['format'] != 1:
        raise ReleaseError('Unsupported release manifest')
    if not isinstance(value['commit'], str) or not SHA.fullmatch(value['commit']):
        raise ReleaseError('Release commit must be immutable')
    if not isinstance(value['migrations'], str) or not re.fullmatch('[0-9a-f]{64}', value['migrations']):
        raise ReleaseError('Missing migration fingerprint')
    for name in ('data_epoch', 'ci_run', 'publish_run'):
        if type(value[name]) is not int or value[name] <= 0:
            raise ReleaseError('Invalid release metadata')
    for name in ('api', 'frontend'):
        prefix = f'ghcr.io/ghalrym/kadan-{name}@'
        if not isinstance(value[name], str) or not value[name].startswith(prefix) or not DIGEST.fullmatch(value[name][len(prefix):]):
            raise ReleaseError('Only official paired image digests are supported')
    return dict(value)


class Releases:
    def __init__(self, fetch=fetch_json):
        self.fetch = fetch

    def verified(self, commit):
        if not SHA.fullmatch(commit):
            raise ReleaseError('Invalid commit')
        release = self.fetch(f'{API}/releases/tags/kadan-{commit}')
        if release.get('draft') or release.get('prerelease') or release.get('tag_name') != f'kadan-{commit}':
            raise ReleaseError('Release is not published')
        # Manifest is a release asset, never a URL supplied by the browser.
        assets = [item for item in release.get('assets', []) if item.get('name') == 'kadan-release.json']
        url = f'https://github.com/{REPOSITORY}/releases/download/kadan-{commit}/kadan-release.json'
        if len(assets) != 1 or assets[0].get('browser_download_url') != url:
            raise ReleaseError('Missing official release manifest')
        manifest = validate_manifest(self.fetch(url))
        if manifest['commit'] != commit:
            raise ReleaseError('Release/image version mismatch')
        for key, path, event in (('ci_run', '.github/workflows/api-tests.yml', 'push'),
                                 ('publish_run', '.github/workflows/release.yml', 'workflow_run')):
            run = self.fetch(f'{API}/actions/runs/{manifest[key]}')
            if (run.get('status') != 'completed' or run.get('conclusion') != 'success'
                    or run.get('head_sha') != commit or run.get('head_branch') != 'master'
                    or run.get('event') != event or run.get('path') != path
                    or run.get('repository', {}).get('full_name') != REPOSITORY):
                raise ReleaseError('Release requires successful exact-master CI and publishing')
        comparison = self.fetch(f'{API}/compare/{commit}...master')
        if comparison.get('status') not in ('ahead', 'identical'):
            raise ReleaseError('Release is not on master')
        return manifest

    def latest(self):
        releases = self.fetch(f'{API}/releases?per_page=20')
        for release in releases:
            tag = release.get('tag_name', '')
            if tag.startswith('kadan-') and SHA.fullmatch(tag[6:]) and not release.get('draft'):
                return self.verified(tag[6:])
        raise ReleaseError('No tested release is published yet')

    def upgrade(self, current, candidate):
        if current['commit'] == candidate['commit']:
            return False
        comparison = self.fetch(f"{API}/compare/{current['commit']}...{candidate['commit']}")
        if comparison.get('status') != 'ahead':
            raise ReleaseError('Automatic checks cannot select a downgrade or divergent release')
        return True


def compatible(current, candidate):
    return all(current[key] == candidate[key] for key in ('migrations', 'data_epoch'))
