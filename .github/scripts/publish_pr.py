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
    (4, 'decisions-api', 'c9d30389c6439bbc83f051702ac839417e586ad9'),
    (5, 'images-api', '6cd11e30342e299605550a777ac3616e07956388'),
    (6, 'video-api', '16b06719d1ce7743fab8c85de8b5dd5072443156'),
    (7, 'speech-api', 'cf46e29b0c2b459ead5dff693b03544f0481b426'),
    (8, 'transcription-api', '33c3a4dbce765118cd38cb4fc5e7420e6ead69d5'),
    (9, 'api-state', 'a6d3b3da8c1ed67755995b19298e526aca4ddc27'),
    (10, 'api-access', '25cb542f3e2e627b88d251d05168d85a96d794f2'),
    (11, 'monitoring-api', 'd64bb9b134fe840cdec0939f0f268340c18f659b'),
)
MIGRATIONS = {}
for _index, (_number, _name, _sha) in enumerate(_MIGRATION_ROWS):
    _previous = _MIGRATION_ROWS[_index - 1] if _index else None
    MIGRATIONS['codex/bot-' + _name] = dict(
        original_number=_number, original_head='codex/' + _name, head_sha=_sha,
        original_base='codex/' + _previous[1] if _previous else 'codex/freetoken-runtime',
        base='codex/bot-' + _previous[1] if _previous else 'codex/freetoken-runtime',
        base_sha=_previous[2] if _previous else 'ce09eb5c77ee88b9120e42eeee0e77c0431dc22c')


def migration(values):
    """Return the fixed migration rule, or None for ordinary master-bound heads.

    Reject unknown reserved aliases and mismatched approved SHAs before API work.
    """
    rule = MIGRATIONS.get(values['head'])
    require(not values['head'].startswith('codex/bot-') or rule is not None,
            'Unapproved migration alias')
    if rule is not None:
        require(values['expected_sha'] == rule['head_sha'], 'Migration SHA is not the approved commit')
    return rule


def target_base(values):
    """Derive the PR base from the reviewed allowlist; callers cannot select it."""
    rule = migration(values)
    return rule['base'] if rule else BASE


class Failure(Exception):
    """Only fixed, sanitized diagnostic strings may reach workflow logs."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Reject redirects so credentials and requests cannot leave the fixed API origin."""
        raise Failure('GitHub API redirect rejected')


def http_diagnostic(path, status, headers):
    """Describe an HTTP result without response bodies, query values, or credentials.

    Only known fixed endpoint paths and bounded decimal rate-limit/retry values
    may reach logs. Unknown paths and nonnumeric header values are omitted.
    """
    endpoint = path.split('?', 1)[0].split('#', 1)[0]
    known = re.fullmatch(re.escape(ROOT) +
        r'(?:/environments/kadan-pr-publishing(?:/deployment-branch-policies)?'
        r'|/pulls(?:/[0-9]+(?:/comments|/reviews)?)?|/issues/[0-9]+/comments)?', endpoint)
    if not known and endpoint != '/installation/repositories':
        endpoint = '<redacted>'
    fields = [f'HTTP {status}', f'path={endpoint}']
    for name in ('x-ratelimit-limit', 'x-ratelimit-remaining', 'x-ratelimit-used',
                 'x-ratelimit-reset', 'retry-after'):
        value = (headers or {}).get(name)
        if isinstance(value, str) and re.fullmatch(r'[0-9]{1,12}', value):
            fields.append(f'{name}={value}')
    return '; '.join(fields)


class API:
    def __init__(self, token=None):
        """Build an API client with optional bearer authentication; perform no request yet."""
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, data=None):
        """Send one fixed-origin JSON request with a timeout and bounded response size.

        Never retry writes; transport/HTTP/JSON errors become sanitized Failure messages.
        A failed write may already have taken effect and must be inspected before retrying.
        """
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
            raise Failure('GitHub API request failed (' +
                          http_diagnostic(path, exc.code, exc.headers) + ')') from None
        except (urllib.error.URLError, OSError, ValueError, RecursionError):
            raise Failure('GitHub API transport or JSON failure') from None


def require(condition, message):
    """Raise Failure on a failed invariant; message must not contain credentials or raw inputs."""
    if not condition:
        raise Failure(message)


def inputs(event, environ):
    """Validate the trusted owner dispatch and return its four string inputs as data.

    Reject wrong repository/ref/actor, unsafe branch names, invalid SHAs, and oversized
    or malformed text before network access. This does not authenticate an App token.
    """
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
    """Yield all list items (or a named list field), failing on malformed or excessive pages.

    Numbered URLs stay on the fixed API origin; untrusted Link targets are ignored.
    """
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
    """Read public metadata to require the expected repository and existing master-only environment.

    Perform no writes or key access. Missing/inaccessible metadata fails closed; this
    check does not prove App authentication or isolate the secret to this workflow.
    """
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
    """Require a named branch ref to point to the expected commit; never update the ref."""
    branch = api.request('GET', ROOT + '/git/ref/heads/' + urllib.parse.quote(values['head'], safe=''))
    require(branch.get('ref') == 'refs/heads/' + values['head'] and
            branch.get('object', {}).get('type') == 'commit' and
            branch.get('object', {}).get('sha') == values['expected_sha'], 'Head branch SHA mismatch')


def pr_url(pr, values):
    """Validate PR repository and head/base identities, then return its canonical GitHub URL."""
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
    """Check pinned alias refs and, by default, their original refs without changing branches."""
    refs = ((values['head'], rule['head_sha']), (rule['base'], rule['base_sha']),
                      (rule['original_head'], rule['head_sha']), (rule['original_base'], rule['base_sha']))
    for head, sha in (refs if originals else refs[:2]):
        branch_sha(api, {'head': head, 'expected_sha': sha})


def migration_guards(api, values, rule):
    """Require an open, unchanged original and no conversation, inline comments, or reviews.

    These read-only checks use existing PR permissions. They are not atomic with
    creation; independent verification remains required before original closure.
    """
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
    """Return an existing matching PR or create one draft with the scoped App token.

    Check the configured installation ID, token repository scope, and exact refs before
    creation. Migration also requires unchanged originals with no discussion. Existing closed
    PRs prevent replacement duplicates. The workflow must serialize calls; creation
    is not atomic with validation, so failed post-write checks report uncertainty
    and never retry. No original PR or branch is mutated.
    """
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
    """Run preflight or publication from the workflow event file and return a process exit code.

    Preflight uses unauthenticated public reads; publication requires the minted token.
    Print only safe status, a validated PR URL, or a sanitized failure; never the key/token.
    """
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
