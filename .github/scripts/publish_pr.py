#!/usr/bin/env python3
"""Restricted draft-PR publisher. Inputs are JSON data, never executable code.

Environment GET does not document can_admins_bypass: administrators must disable
bypass in the UI. When the API does return it, an enabled bypass is rejected.
The workflow must serialize publishing runs and protect this script on master.
"""
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY = 'ghalrym/Kadan'
REPOSITORY_ID = 1403845248
INSTALLATION_ID = '167899795'
OWNER = 'ghalrym'
OWNER_ID = 177494187
BASE = 'master'
ENVIRONMENT = 'kadan-pr-publishing'
ROOT = '/repos/' + REPOSITORY

# One-time, reviewed migration only. No dispatch input can choose a base or SHA.
_MIGRATION_ROWS = (
    (4, 'decisions-api', '25c08620276d0fc6115e555e7e1d1d2c45982ff1'),
    (5, 'images-api', 'c6e8428d6e5c87ff56c47b8602855f755a1b3b90'),
    (6, 'video-api', '6da2a4b0f5c80727eb8a9d1d6596e8b1bdd88ac2'),
    (7, 'speech-api', 'cd4f7098e1cb0908049b89ca370e013f600684c6'),
    (8, 'transcription-api', '7cdc6d4ce510ffd16e7feec97bc5621b8c58bf93'),
    (9, 'api-state', 'bee7b466f9cc8bb030f76caee908060b96902759'),
    (10, 'api-access', '02a533608155e4669227c89b91d1652fb900f17e'),
    (11, 'monitoring-api', '1d1cb1f60f18ab9fb848dc8c5341899bffc45a4b'),
)
MIGRATIONS = {}
for _index, (_number, _name, _sha) in enumerate(_MIGRATION_ROWS):
    _previous = _MIGRATION_ROWS[_index - 1] if _index else None
    MIGRATIONS['codex/bot-' + _name] = dict(
        original_number=_number, original_head='codex/' + _name, head_sha=_sha,
        original_base='codex/' + _previous[1] if _previous else 'codex/freetoken-runtime',
        base='codex/bot-' + _previous[1] if _previous else 'codex/freetoken-runtime',
        base_sha=_previous[2] if _previous else 'd2773d4241a264f84d63235650a226c434df47aa')


def migration(values):
    rule = MIGRATIONS.get(values['head'])
    require(not values['head'].startswith('codex/bot-') or rule is not None,
            'Unapproved migration alias')
    if rule is not None:
        require(values['expected_sha'] == rule['head_sha'], 'Migration SHA is not the approved commit')
    return rule


def target_base(values):
    rule = migration(values)
    return rule['base'] if rule else BASE


class Failure(Exception):
    """Only fixed, sanitized diagnostic strings may reach workflow logs."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Failure('GitHub API redirect rejected')


class API:
    def __init__(self, token=None):
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, data=None):
        if not path.startswith('/') or path.startswith('//'):
            raise Failure('Invalid API path')
        headers = {'Accept': 'application/vnd.github+json',
                   'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'Kadan-PR-publisher'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        body = None if data is None else json.dumps(data).encode('utf-8')
        if body is not None:
            headers['Content-Type'] = 'application/json'
        request = urllib.request.Request('https://api.github.com' + path,
                                         data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=30) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise Failure('GitHub API response exceeded size limit')
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise Failure(f'GitHub API request failed (HTTP {exc.code})') from None
        except (urllib.error.URLError, OSError, ValueError, RecursionError):
            raise Failure('GitHub API transport or JSON failure') from None


def require(condition, message):
    if not condition:
        raise Failure(message)


def inputs(event, environ):
    require(environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch', 'Manual dispatch required')
    require(environ.get('GITHUB_REPOSITORY') == REPOSITORY and
            environ.get('GITHUB_REPOSITORY_ID') == str(REPOSITORY_ID), 'Unexpected workflow repository')
    require(environ.get('GITHUB_REF') == 'refs/heads/master', 'Workflow must run from master')
    require(environ.get('GITHUB_ACTOR') == OWNER and
            environ.get('GITHUB_ACTOR_ID') == str(OWNER_ID) and
            environ.get('GITHUB_TRIGGERING_ACTOR') == OWNER, 'Manual owner dispatch required')
    sender = event.get('sender', {})
    require(sender.get('login') == OWNER and sender.get('id') == OWNER_ID,
            'Unexpected dispatch sender')
    repo = event.get('repository', {})
    require(repo.get('id') == REPOSITORY_ID and repo.get('full_name') == REPOSITORY,
            'Unexpected event repository')
    values = event.get('inputs', {})
    require(isinstance(values, dict) and set(values) == {'head', 'title', 'body', 'expected_sha'},
            'Expected exactly head, title, body and expected_sha inputs')
    require(all(isinstance(value, str) for value in values.values()), 'Inputs must be strings')
    head = values['head']
    require(len(head) <= 120 and re.fullmatch(r'codex/[a-z0-9][a-z0-9._-]*', head)
            and '..' not in head and '.lock' not in head and not head.endswith('.'),
            'Invalid codex branch name')
    require(re.fullmatch(r'[0-9a-f]{40}', values['expected_sha']), 'Expected SHA must be 40 lowercase hex characters')
    require(1 <= len(values['title']) <= 256 and values['title'].strip()
            and all(ord(c) >= 32 and ord(c) != 127 for c in values['title']), 'Invalid PR title')
    require(len(values['body']) <= 60000 and all(ord(c) >= 32 or c in '\n\r\t' for c in values['body'])
            and '\x7f' not in values['body'], 'Invalid PR body')
    # Reject unpaired Unicode surrogates before any network request.
    try:
        json.dumps(values, ensure_ascii=False).encode('utf-8')
    except UnicodeError:
        raise Failure('Invalid input Unicode') from None
    migration(values)
    return values


def pages(api, path, key=None):
    """Numbered URLs stay on the fixed API origin; do not trust Link targets."""
    for page in range(1, 1001):
        separator = '&' if '?' in path else '?'
        result = api.request('GET', f'{path}{separator}per_page=100&page={page}')
        items = result.get(key) if key and isinstance(result, dict) else result if key is None else None
        require(isinstance(items, list), 'Malformed paginated GitHub response')
        yield from items
        if len(items) < 100:
            return
    raise Failure('GitHub pagination safety limit exceeded')


def preflight(api):
    repo = api.request('GET', ROOT)
    require(repo.get('id') == REPOSITORY_ID and repo.get('full_name') == REPOSITORY
            and repo.get('private') is False and repo.get('default_branch') == BASE,
            'Repository identity, visibility or default branch mismatch')
    env = api.request('GET', ROOT + '/environments/' + ENVIRONMENT)
    require(env.get('name') == ENVIRONMENT, 'Protected environment does not exist')
    policy = env.get('deployment_branch_policy') or {}
    require(policy.get('custom_branch_policies') is True and policy.get('protected_branches') is False,
            'Environment requires custom deployment branch policies')
    rules = env.get('protection_rules')
    require(isinstance(rules, list), 'Environment protection rules unavailable')
    # Owner dispatch authorizes publishing; a second deployment reviewer is not required.
    # Existing environment review rules, if configured, still apply in GitHub.
    require('can_admins_bypass' not in env or env['can_admins_bypass'] is False,
            'Environment administrator bypass must be disabled')
    policies = list(pages(api, ROOT + '/environments/' + ENVIRONMENT + '/deployment-branch-policies', 'branch_policies'))
    require(len(policies) == 1 and policies[0].get('name') == BASE and policies[0].get('type') == 'branch',
            'Environment must allow exactly the master branch and no tags')


def branch_sha(api, values):
    branch = api.request('GET', ROOT + '/git/ref/heads/' + urllib.parse.quote(values['head'], safe=''))
    require(branch.get('ref') == 'refs/heads/' + values['head'] and
            branch.get('object', {}).get('type') == 'commit' and
            branch.get('object', {}).get('sha') == values['expected_sha'], 'Head branch SHA mismatch')


def pr_url(pr, values):
    require(pr.get('head', {}).get('ref') == values['head'] and pr.get('base', {}).get('ref') == target_base(values)
            and pr.get('head', {}).get('repo', {}).get('id') == REPOSITORY_ID
            and pr.get('base', {}).get('repo', {}).get('id') == REPOSITORY_ID,
            'Pull request repository or branch mismatch')
    number = pr.get('number')
    require(type(number) is int and number > 0, 'Invalid pull request number')
    url = f'https://github.com/{REPOSITORY}/pull/{number}'
    require(pr.get('html_url') == url, 'Invalid pull request URL')
    return url


def migration_refs(api, values, rule, originals=True):
    refs = ((values['head'], rule['head_sha']), (rule['base'], rule['base_sha']),
                      (rule['original_head'], rule['head_sha']), (rule['original_base'], rule['base_sha']))
    for head, sha in (refs if originals else refs[:2]):
        branch_sha(api, {'head': head, 'expected_sha': sha})


def migration_guards(api, values, rule):
    """PR-read permission suffices; any discussion blocks migration."""
    migration_refs(api, values, rule)
    original = api.request('GET', ROOT + '/pulls/' + str(rule['original_number']))
    require(original.get('number') == rule['original_number'] and original.get('state') == 'open'
            and original.get('merged_at') is None
            and original.get('head', {}).get('ref') == rule['original_head']
            and original.get('head', {}).get('sha') == rule['head_sha']
            and original.get('base', {}).get('ref') == rule['original_base']
            and original.get('base', {}).get('sha') == rule['base_sha']
            and original.get('head', {}).get('repo', {}).get('id') == REPOSITORY_ID
            and original.get('base', {}).get('repo', {}).get('id') == REPOSITORY_ID,
            'Original pull request changed; migration denied')
    number = str(rule['original_number'])
    for path in ('/issues/' + number + '/comments', '/pulls/' + number + '/comments',
                 '/pulls/' + number + '/reviews'):
        require(not list(pages(api, ROOT + path)), 'Original pull request has discussion or reviews; migration denied')


def publish(api, values, environ):
    rule = migration(values)
    base = target_base(values)
    require(environ.get('APP_INSTALLATION_ID') == INSTALLATION_ID, 'Unexpected App installation')
    installation = api.request('GET', '/installation/repositories?per_page=100&page=1')
    repositories = installation.get('repositories')
    require(installation.get('total_count') == 1 and isinstance(repositories, list) and len(repositories) == 1 and repositories[0].get('id') == REPOSITORY_ID
            and repositories[0].get('full_name') == REPOSITORY, 'App token must access only Kadan')
    branch_sha(api, values)
    if rule:
        migration_refs(api, values, rule, originals=False)
    query = urllib.parse.urlencode({'state': 'all', 'head': 'ghalrym:' + values['head'], 'base': base})
    existing = list(pages(api, ROOT + '/pulls?' + query))
    if existing:
        # Closed and merged PRs count too: reruns never create replacements.
        urls = [pr_url(pr, values) for pr in existing]
        if rule:
            require(all(pr.get('head', {}).get('sha') == rule['head_sha']
                        and pr.get('base', {}).get('sha') == rule['base_sha'] for pr in existing),
                    'Existing replacement SHA mismatch')
        return urls[0]
    if rule:
        migration_guards(api, values, rule)
    body = values['body']
    if rule:
        body = (f"Replacement draft for original PR https://github.com/{REPOSITORY}/pull/{rule['original_number']}. "
                'This changes PR authorship only; commit authorship and approved head/base commits are unchanged.\n\n' + body)
    branch_sha(api, values)  # Narrow the race between listing and creation.
    if rule:
        migration_refs(api, values, rule, originals=False)
    try:
        created = api.request('POST', ROOT + '/pulls', {
            'head': values['head'], 'base': base, 'title': values['title'], 'body': body,
            'draft': True, 'maintainer_can_modify': False})
        url = pr_url(created, values)
        require(created.get('draft') is True, 'Created pull request is not a draft')
        verified = api.request('GET', ROOT + '/pulls/' + str(created['number']))
        require(pr_url(verified, values) == url and verified.get('head', {}).get('sha') == values['expected_sha'],
                'Created pull request SHA could not be verified')
        branch_sha(api, values)
        if rule:
            require(verified.get('base', {}).get('sha') == rule['base_sha'], 'Replacement base SHA mismatch')
            migration_guards(api, values, rule)
        return url
    except (Failure, ValueError, TypeError, AttributeError, KeyError, RecursionError):
        raise Failure('Publish outcome uncertain; inspect existing pull requests and branch SHA. No creation retry was attempted.') from None


def main(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    argv = sys.argv[1:] if argv is None else argv
    try:
        require(argv in (['preflight'], ['publish']), 'Usage: publish_pr.py preflight|publish')
        path = environ.get('GITHUB_EVENT_PATH')
        require(bool(path), 'Missing workflow event path')
        raw = Path(path).read_bytes()
        require(len(raw) <= 1024 * 1024, 'Workflow event exceeds size limit')
        values = inputs(json.loads(raw), environ)
        if argv == ['preflight']:
            preflight(API())  # Public metadata only; no credential or settings mutation.
            print('Owner dispatch and master-only environment validated; verify administrator bypass is disabled in the environment UI.')
        else:
            token = environ.get('GH_TOKEN')
            require(bool(token), 'Missing installation token')
            url = publish(API(token), values, environ)
            print(url)
        return 0
    except Failure as exc:
        print(str(exc), file=sys.stderr)
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
        print('Invalid workflow data or API response', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
