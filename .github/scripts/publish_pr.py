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
BASE = 'master'
ENVIRONMENT = 'kadan-pr-publishing'
ROOT = '/repos/' + REPOSITORY


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
    reviewers = [rule for rule in rules if isinstance(rule, dict) and rule.get('type') == 'required_reviewers']
    require(len(reviewers) == 1 and reviewers[0].get('prevent_self_review') is True,
            'Environment requires reviewers and prevention of self-review')
    people = reviewers[0].get('reviewers')
    require(isinstance(people, list) and len(people) >= 1 and all(
        isinstance(person, dict) and person.get('type') in ('User', 'Team')
        and isinstance(person.get('reviewer'), dict)
        and type(person['reviewer'].get('id')) is int and person['reviewer']['id'] > 0 for person in people),
        'Environment requires at least one valid reviewer')
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
    require(pr.get('head', {}).get('ref') == values['head'] and pr.get('base', {}).get('ref') == BASE
            and pr.get('head', {}).get('repo', {}).get('id') == REPOSITORY_ID
            and pr.get('base', {}).get('repo', {}).get('id') == REPOSITORY_ID,
            'Pull request repository or branch mismatch')
    number = pr.get('number')
    require(type(number) is int and number > 0, 'Invalid pull request number')
    url = f'https://github.com/{REPOSITORY}/pull/{number}'
    require(pr.get('html_url') == url, 'Invalid pull request URL')
    return url


def publish(api, values, environ):
    require(environ.get('APP_INSTALLATION_ID') == INSTALLATION_ID, 'Unexpected App installation')
    installation = api.request('GET', '/installation/repositories?per_page=100&page=1')
    repositories = installation.get('repositories')
    require(installation.get('total_count') == 1 and isinstance(repositories, list) and len(repositories) == 1 and repositories[0].get('id') == REPOSITORY_ID
            and repositories[0].get('full_name') == REPOSITORY, 'App token must access only Kadan')
    branch_sha(api, values)
    query = urllib.parse.urlencode({'state': 'all', 'head': 'ghalrym:' + values['head'], 'base': BASE})
    existing = list(pages(api, ROOT + '/pulls?' + query))
    if existing:
        # Closed and merged PRs count too: reruns never create replacements.
        urls = [pr_url(pr, values) for pr in existing]
        return urls[0]
    branch_sha(api, values)  # Narrow the race between listing and creation.
    try:
        created = api.request('POST', ROOT + '/pulls', {
            'head': values['head'], 'base': BASE, 'title': values['title'], 'body': values['body'],
            'draft': True, 'maintainer_can_modify': False})
        url = pr_url(created, values)
        require(created.get('draft') is True, 'Created pull request is not a draft')
        verified = api.request('GET', ROOT + '/pulls/' + str(created['number']))
        require(pr_url(verified, values) == url and verified.get('head', {}).get('sha') == values['expected_sha'],
                'Created pull request SHA could not be verified')
        branch_sha(api, values)
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
            print('Preflight passed; administrator bypass must be disabled in the environment UI.')
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
