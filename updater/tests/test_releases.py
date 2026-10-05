import copy
import unittest

from updater.releases import API, Releases, ReleaseError, compatible, validate_manifest


def manifest(commit='a' * 40):
    return dict(format=1, commit=commit, api='ghcr.io/ghalrym/kadan-api@sha256:' + '1' * 64,
                frontend='ghcr.io/ghalrym/kadan-frontend@sha256:' + '2' * 64,
                migrations='3' * 64, data_epoch=1, ci_run=101, publish_run=102)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.value = manifest()
        sha = self.value['commit']
        self.asset = f'https://github.com/ghalrym/Kadan/releases/download/kadan-{sha}/kadan-release.json'
        self.release = {'tag_name': 'kadan-' + sha, 'draft': False, 'prerelease': False,
                        'assets': [{'name': 'kadan-release.json', 'browser_download_url': self.asset}]}
        self.records = {f'{API}/releases/tags/kadan-{sha}': self.release, self.asset: self.value,
                        f'{API}/compare/{sha}...master': {'status': 'ahead'}}
        for number, path, event in ((101, 'api-tests.yml', 'push'), (102, 'release.yml', 'workflow_run')):
            self.records[f'{API}/actions/runs/{number}'] = dict(status='completed', conclusion='success',
                head_sha=sha, head_branch='master', event=event, path='.github/workflows/' + path,
                repository={'full_name': 'ghalrym/Kadan'})
        self.source = Releases(lambda url: copy.deepcopy(self.records[url]))

    def test_accepts_only_paired_digest_manifest(self):
        self.assertEqual(self.source.verified(self.value['commit']), self.value)
        for key, value in (('commit', 'master'), ('api', 'ghcr.io/ghalrym/kadan-api:latest'),
                           ('frontend', 'other@sha256:' + '2' * 64), ('ci_run', True)):
            with self.subTest(key=key), self.assertRaises(ReleaseError):
                validate_manifest({**self.value, key: value})

    def test_failed_running_untrusted_and_mismatched_ci_are_rejected(self):
        for number in (101, 102):
            run = self.records[f'{API}/actions/runs/{number}']
            for key, value in (('conclusion', 'failure'), ('status', 'in_progress'), ('head_sha', 'b' * 40),
                               ('head_branch', 'feature'), ('path', '.github/workflows/other.yml'),
                               ('repository', {'full_name': 'attacker/Kadan'})):
                with self.subTest(number=number, key=key):
                    previous = run[key]
                    run[key] = value
                    with self.assertRaises(ReleaseError):
                        self.source.verified(self.value['commit'])
                    run[key] = previous

    def test_draft_manifest_swap_and_untrusted_asset_fail_closed(self):
        self.release['draft'] = True
        with self.assertRaises(ReleaseError): self.source.verified(self.value['commit'])
        self.release['draft'] = False
        self.records[self.asset] = manifest('b' * 40)
        with self.assertRaises(ReleaseError): self.source.verified(self.value['commit'])
        self.records[self.asset] = self.value
        self.release['assets'][0]['browser_download_url'] = 'http://localhost/internal'
        with self.assertRaises(ReleaseError): self.source.verified(self.value['commit'])

    def test_migration_and_data_epoch_changes_require_manual_upgrade(self):
        self.assertTrue(compatible(self.value, manifest('b' * 40)))
        for key, value in (('migrations', '4' * 64), ('data_epoch', 2)):
            self.assertFalse(compatible(self.value, {**manifest('b' * 40), key: value}))

    def test_downgrade_and_divergence_are_not_offered(self):
        candidate = manifest('b' * 40)
        endpoint = f'{API}/compare/{self.value["commit"]}...{candidate["commit"]}'
        for status in ('behind', 'diverged'):
            self.records[endpoint] = {'status': status}
            with self.assertRaises(ReleaseError): self.source.upgrade(self.value, candidate)
