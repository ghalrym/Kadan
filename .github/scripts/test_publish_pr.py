"""No network or credentials: exercise publication boundaries with API doubles."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('publisher', Path(__file__).with_name('publish_pr.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class FakeAPI:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def request(self, method, path, data=None):
        self.calls.append((method, path, data))
        if not self.answers:
            raise AssertionError('Unexpected API call')
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return copy.deepcopy(answer)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.values = dict(head='codex/test-feature', title='A safe title', body='Body\n`literal` $(data)', expected_sha='a' * 40)
        self.environ = dict(GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_REPOSITORY=p.REPOSITORY,
                            GITHUB_REPOSITORY_ID=str(p.REPOSITORY_ID), GITHUB_REF='refs/heads/master',
                            APP_INSTALLATION_ID=p.INSTALLATION_ID, GITHUB_ACTOR='ghalrym',
                            GITHUB_ACTOR_ID='177494187', GITHUB_TRIGGERING_ACTOR='ghalrym')
        self.repo = dict(id=p.REPOSITORY_ID, full_name=p.REPOSITORY, private=False, default_branch='master')
        self.event = dict(repository=self.repo, inputs=self.values,
                          sender=dict(login='ghalrym', id=177494187))
        self.environment = dict(name=p.ENVIRONMENT,
            deployment_branch_policy=dict(custom_branch_policies=True, protected_branches=False),
            protection_rules=[dict(type='branch_policy')])
        self.policy = dict(branch_policies=[dict(name='master', type='branch')])
        self.branch = dict(ref='refs/heads/' + self.values['head'], object=dict(type='commit', sha='a' * 40))
        self.installation = dict(total_count=1, repositories=[self.repo])
        self.pr = dict(number=42, html_url='https://github.com/ghalrym/Kadan/pull/42', draft=True,
                       head=dict(ref=self.values['head'], sha='a' * 40, repo=self.repo),
                       base=dict(ref='master', repo=self.repo))

    def test_valid_inputs_are_data(self):
        self.assertEqual(p.inputs(self.event, self.environ), self.values)

    def test_reject_invalid_branch_names(self):
        for name in ('master', 'codex/../foo', 'codex/a.lock', 'codex/a.lock-x', 'codex/a.',
                     'codex/A', 'codex/a/b', 'codex/-a', 'codex/x\n', 'codex/' + 'a' * 120):
            with self.subTest(name=name), self.assertRaises(p.Failure):
                p.inputs(dict(self.event, inputs=dict(self.values, head=name)), self.environ)

    def test_reject_sha_and_content_limits(self):
        for key, value in [('expected_sha', 'A' * 40), ('expected_sha', 'a' * 39),
                           ('title', ''), ('title', 'x' * 257), ('title', 'x\ny'),
                           ('body', 'x' * 60001), ('body', '\0'), ('body', '\ud800')]:
            with self.subTest(key=key), self.assertRaises(p.Failure):
                p.inputs(dict(self.event, inputs=dict(self.values, **{key: value})), self.environ)

    def test_wrong_workflow_identity_or_ref_rejected(self):
        for key in ('GITHUB_REF', 'GITHUB_EVENT_NAME', 'GITHUB_REPOSITORY', 'GITHUB_REPOSITORY_ID'):
            with self.subTest(key=key), self.assertRaises(p.Failure):
                p.inputs(self.event, dict(self.environ, **{key: 'wrong'}))
        with self.assertRaises(p.Failure):
            p.inputs(dict(self.event, repository=dict(self.repo, id=1)), self.environ)

    def test_owner_identity_required(self):
        for key in ('GITHUB_ACTOR', 'GITHUB_ACTOR_ID', 'GITHUB_TRIGGERING_ACTOR'):
            for value in ('other', '', None):
                with self.subTest(key=key, value=value), self.assertRaises(p.Failure):
                    p.inputs(self.event, dict(self.environ, **{key: value}))
        for sender in ({}, dict(login='other', id=177494187), dict(login='ghalrym', id=1)):
            with self.subTest(sender=sender), self.assertRaises(p.Failure):
                p.inputs(dict(self.event, sender=sender), self.environ)

    def test_denied_owner_never_reaches_api(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'event.json'
            path.write_text(json.dumps(self.event))
            for mode in ('preflight', 'publish'):
                env = dict(self.environ, GITHUB_EVENT_PATH=str(path), GITHUB_ACTOR_ID='1')
                with patch.object(p, 'API') as api, patch('sys.stderr', io.StringIO()):
                    self.assertEqual(p.main([mode], env), 1)
                    api.assert_not_called()

    def test_sole_owner_preflight_without_review_gate(self):
        self.assertEqual(p.inputs(self.event, self.environ), self.values)
        api = FakeAPI([self.repo, dict(self.environment, protection_rules=[]), self.policy])
        p.preflight(api)
        self.assertTrue(all(call[0] == 'GET' for call in api.calls))

    def test_unknown_inputs_rejected(self):
        with self.assertRaises(p.Failure):
            p.inputs(dict(self.event, inputs=dict(self.values, base='other')), self.environ)

    def test_preflight_public_read_only(self):
        api = FakeAPI([self.repo, self.environment, self.policy])
        p.preflight(api)
        self.assertTrue(all(call[0] == 'GET' for call in api.calls))

    def test_repository_guards(self):
        for change in (dict(id=1), dict(private=True), dict(default_branch='main'), dict(full_name='other/Kadan')):
            api = FakeAPI([dict(self.repo, **change)])
            with self.assertRaises(p.Failure):
                p.preflight(api)
            self.assertEqual(len(api.calls), 1)

    def test_missing_environment_is_not_created(self):
        api = FakeAPI([self.repo, p.Failure('GitHub API request failed (HTTP 404)')])
        with self.assertRaises(p.Failure):
            p.preflight(api)
        self.assertTrue(all(call[0] == 'GET' for call in api.calls))

    def test_policy_guards(self):
        for mutate in (
            lambda e: e.update(protection_rules=None),
            lambda e: e.update(deployment_branch_policy=None),
            lambda e: e.update(can_admins_bypass=True),
            lambda e: e.update(can_admins_bypass=None),
        ):
            env = copy.deepcopy(self.environment)
            mutate(env)
            with self.assertRaises(p.Failure):
                p.preflight(FakeAPI([self.repo, env, self.policy]))

    def test_branch_rules_exact_master_only(self):
        for policies in ([], [dict(name='*', type='branch')], [dict(name='master', type='tag')],
                         [dict(name='master', type='branch'), dict(name='codex/*', type='branch')]):
            with self.subTest(policies=policies), self.assertRaises(p.Failure):
                p.preflight(FakeAPI([self.repo, self.environment, dict(branch_policies=policies)]))

    def test_branch_rules_pagination(self):
        api = FakeAPI([self.repo, self.environment, dict(branch_policies=[dict(name='master', type='branch')] * 100),
                       dict(branch_policies=[])])
        with self.assertRaises(p.Failure):
            p.preflight(api)
        self.assertIn('page=2', api.calls[-1][1])

    def test_installation_guards_before_mutation(self):
        api = FakeAPI([])
        with self.assertRaises(p.Failure):
            p.publish(api, self.values, dict(self.environ, APP_INSTALLATION_ID='1'))
        self.assertEqual(api.calls, [])
        for repos in ([], [dict(self.repo, id=1)], [self.repo, dict(self.repo, id=2)]):
            api = FakeAPI([dict(repositories=repos)])
            with self.assertRaises(p.Failure):
                p.publish(api, self.values, self.environ)
            self.assertEqual(len(api.calls), 1)

    def test_inconsistent_installation_total_rejected(self):
        api = FakeAPI([dict(total_count=2, repositories=[self.repo])])
        with self.assertRaises(p.Failure):
            p.publish(api, self.values, self.environ)
        self.assertEqual(len(api.calls), 1)

    def test_sha_guard(self):
        api = FakeAPI([self.installation, dict(self.branch, object=dict(type='commit', sha='b' * 40))])
        with self.assertRaises(p.Failure):
            p.publish(api, self.values, self.environ)
        self.assertTrue(all(call[0] == 'GET' for call in api.calls))

    def test_existing_open_closed_merged_never_mutated(self):
        for state in ('open', 'closed'):
            api = FakeAPI([self.installation, self.branch, [dict(self.pr, state=state)]])
            self.assertEqual(p.publish(api, self.values, self.environ), self.pr['html_url'])
            self.assertTrue(all(call[0] == 'GET' for call in api.calls))
            self.assertIn('state=all', api.calls[-1][1])

    def test_existing_pr_pagination(self):
        api = FakeAPI([self.installation, self.branch, [self.pr] * 100, [self.pr]])
        self.assertEqual(p.publish(api, self.values, self.environ), self.pr['html_url'])
        self.assertIn('page=2', api.calls[-1][1])
        self.assertTrue(all(call[0] == 'GET' for call in api.calls))

    def test_wrong_pr_identity_fails_closed(self):
        api = FakeAPI([self.installation, self.branch, [dict(self.pr, html_url='https://evil.example/')]])
        with self.assertRaises(p.Failure):
            p.publish(api, self.values, self.environ)

    def test_create_draft_once_and_verify(self):
        api = FakeAPI([self.installation, self.branch, [], self.branch, self.pr, self.pr, self.branch])
        self.assertEqual(p.publish(api, self.values, self.environ), self.pr['html_url'])
        writes = [call for call in api.calls if call[0] != 'GET']
        self.assertEqual(len(writes), 1)
        self.assertTrue(writes[0][2]['draft'])
        self.assertEqual(writes[0][2]['body'], self.values['body'])
        self.assertFalse(writes[0][2]['maintainer_can_modify'])

    def test_ambiguous_post_and_changed_sha_never_retry(self):
        cases = ([p.Failure('timeout')], [self.pr, dict(self.pr, head=dict(self.pr['head'], sha='b' * 40))],
                 [self.pr, p.Failure('HTTP 503')], [None], [self.pr, None])
        for tail in cases:
            api = FakeAPI([self.installation, self.branch, [], self.branch] + tail)
            with self.assertRaisesRegex(p.Failure, 'outcome uncertain'):
                p.publish(api, self.values, self.environ)
            self.assertEqual(sum(call[0] == 'POST' for call in api.calls), 1)

    def test_changed_branch_during_listing_prevents_creation(self):
        api = FakeAPI([self.installation, self.branch, [], dict(self.branch, object={})])
        with self.assertRaises(p.Failure):
            p.publish(api, self.values, self.environ)
        self.assertFalse(any(call[0] == 'POST' for call in api.calls))

    def test_redirect_rejected(self):
        with self.assertRaises(p.Failure):
            p.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://evil.example')

    def test_cli_never_logs_body_or_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'event.json'
            path.write_text(json.dumps(self.event))
            env = dict(self.environ, GITHUB_EVENT_PATH=str(path), GH_TOKEN='secret-test-token')
            api = FakeAPI([self.installation, self.branch, [], self.branch, p.Failure('HTTP 422')])
            out, err = io.StringIO(), io.StringIO()
            with patch.object(p, 'API', return_value=api), patch('sys.stdout', out), patch('sys.stderr', err):
                self.assertEqual(p.main(['publish'], env), 1)
            self.assertNotIn(env['GH_TOKEN'], out.getvalue() + err.getvalue())
            self.assertNotIn(self.values['body'], out.getvalue() + err.getvalue())



class MigrationTests(unittest.TestCase):
    def setUp(self):
        PublisherTests.setUp(self)
        self.values.update(head='codex/bot-images-api', expected_sha=p.MIGRATIONS['codex/bot-images-api']['head_sha'])
        self.rule = p.MIGRATIONS[self.values['head']]
        self.original = dict(number=5, state='open', merged_at=None,
            head=dict(ref=self.rule['original_head'], sha=self.rule['head_sha'], repo=self.repo),
            base=dict(ref=self.rule['original_base'], sha=self.rule['base_sha'], repo=self.repo))
        self.pr['head'].update(ref=self.values['head'], sha=self.rule['head_sha'])
        self.pr['base'].update(ref=self.rule['base'], sha=self.rule['base_sha'])
        self.existing = []
        self.comments = {'/issues/5/comments': [], '/pulls/5/comments': [], '/pulls/5/reviews': []}
        self.refs = {self.values['head']: self.rule['head_sha'], self.rule['base']: self.rule['base_sha'],
                     self.rule['original_head']: self.rule['head_sha'], self.rule['original_base']: self.rule['base_sha']}
        self.calls = []
        self.after_post = None
        test = self
        class API:
            def request(self, method, path, data=None):
                test.calls.append((method, path, data))
                parsed = p.urllib.parse.urlsplit(path)
                path = parsed.path
                if method == 'POST':
                    test.assertEqual(path, p.ROOT + '/pulls')
                    if test.after_post:
                        test.after_post()
                    return copy.deepcopy(test.pr)
                test.assertEqual(method, 'GET')
                if path == '/installation/repositories':
                    return test.installation
                if path.startswith(p.ROOT + '/git/ref/heads/'):
                    name = p.urllib.parse.unquote(path.removeprefix(p.ROOT + '/git/ref/heads/'))
                    return dict(ref='refs/heads/' + name, object=dict(type='commit', sha=test.refs.get(name)))
                if path == p.ROOT + '/pulls':
                    query = p.urllib.parse.parse_qs(parsed.query)
                    test.assertEqual(query['base'], [test.rule['base']])
                    test.assertEqual(query['state'], ['all'])
                    return copy.deepcopy(test.existing)
                if path == p.ROOT + '/pulls/' + str(test.rule['original_number']):
                    return copy.deepcopy(test.original)
                if path == p.ROOT + '/pulls/42':
                    return copy.deepcopy(test.pr)
                if path.removeprefix(p.ROOT) in test.comments:
                    return copy.deepcopy(test.comments[path.removeprefix(p.ROOT)])
                raise AssertionError('Unexpected API path: ' + path)
        self.api = API()

    def test_happy_migration_fixed_base_transparent_body_single_draft(self):
        self.assertEqual(p.publish(self.api, self.values, self.environ), self.pr['html_url'])
        writes = [call for call in self.calls if call[0] != 'GET']
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][2]['base'], 'codex/bot-decisions-api')
        self.assertTrue(writes[0][2]['draft'])
        self.assertTrue(writes[0][2]['body'].startswith('Replacement draft for original PR https://github.com/ghalrym/Kadan/pull/5.'))
        self.assertTrue(writes[0][2]['body'].endswith(self.values['body']))

    def test_first_and_last_migration_create_with_pinned_bases(self):
        for alias in ('codex/bot-decisions-api', 'codex/bot-monitoring-api'):
            with self.subTest(alias=alias):
                self.setUp()
                self.rule = p.MIGRATIONS[alias]
                self.values.update(head=alias, expected_sha=self.rule['head_sha'])
                number = self.rule['original_number']
                self.original.update(number=number)
                self.original['head'].update(ref=self.rule['original_head'], sha=self.rule['head_sha'])
                self.original['base'].update(ref=self.rule['original_base'], sha=self.rule['base_sha'])
                self.pr['head'].update(ref=alias, sha=self.rule['head_sha'])
                self.pr['base'].update(ref=self.rule['base'], sha=self.rule['base_sha'])
                self.refs = {alias: self.rule['head_sha'], self.rule['base']: self.rule['base_sha'],
                             self.rule['original_head']: self.rule['head_sha'], self.rule['original_base']: self.rule['base_sha']}
                self.comments = {f'/issues/{number}/comments': [], f'/pulls/{number}/comments': [], f'/pulls/{number}/reviews': []}
                self.assertEqual(p.publish(self.api, self.values, self.environ), self.pr['html_url'])
                posts = [call for call in self.calls if call[0] == 'POST']
                self.assertEqual(len(posts), 1)
                self.assertEqual(posts[0][2]['base'], self.rule['base'])
                self.assertIn(f'/pull/{number}.', posts[0][2]['body'])

    def test_all_eight_rules_have_fixed_approved_chain(self):
        self.assertEqual(len(p.MIGRATIONS), 8)
        previous_base, previous_sha = 'codex/freetoken-runtime', 'ce09eb5c77ee88b9120e42eeee0e77c0431dc22c'
        for alias, rule in p.MIGRATIONS.items():
            self.assertEqual(rule['base'], previous_base)
            self.assertEqual(rule['base_sha'], previous_sha)
            self.assertEqual(p.target_base(dict(self.values, head=alias, expected_sha=rule['head_sha'])), previous_base)
            previous_base, previous_sha = alias, rule['head_sha']

    def test_unapproved_alias_or_sha_rejected_before_network(self):
        for values in (dict(self.values, head='codex/bot-unknown'), dict(self.values, expected_sha='a' * 40)):
            with self.assertRaises(p.Failure):
                p.inputs(dict(self.event, inputs=values), self.environ)
            with self.assertRaises(p.Failure):
                p.publish(self.api, values, self.environ)
        self.assertEqual(self.calls, [])

    def test_any_original_or_alias_ref_change_blocks_creation(self):
        for name in self.refs:
            with self.subTest(name=name):
                before = self.refs[name]
                self.refs[name] = 'b' * 40
                with self.assertRaises(p.Failure):
                    p.publish(self.api, self.values, self.environ)
                self.refs[name] = before
                self.assertFalse(any(call[0] == 'POST' for call in self.calls))

    def test_original_state_identity_or_sha_change_blocks_creation(self):
        changes = [('state', 'closed'), ('number', 6), ('merged_at', 'date')]
        for key, value in changes:
            before = copy.deepcopy(self.original)
            self.original[key] = value
            with self.assertRaises(p.Failure):
                p.publish(self.api, self.values, self.environ)
            self.original = before
        for side in ('head', 'base'):
            for key, value in [('ref', 'wrong'), ('sha', 'b' * 40), ('repo', {'id': 1})]:
                before = copy.deepcopy(self.original)
                self.original[side][key] = value
                with self.assertRaises(p.Failure):
                    p.publish(self.api, self.values, self.environ)
                self.original = before
        self.assertFalse(any(call[0] == 'POST' for call in self.calls))

    def test_each_discussion_channel_denies_creation(self):
        for path in self.comments:
            self.comments[path] = [{'id': 123}]
            with self.assertRaisesRegex(p.Failure, 'discussion or reviews'):
                p.publish(self.api, self.values, self.environ)
            self.comments[path] = []
        self.assertFalse(any(call[0] == 'POST' for call in self.calls))

    def test_existing_replacement_after_original_closed_is_read_only(self):
        self.existing = [dict(self.pr, state='closed')]
        self.original['state'] = 'closed'
        self.comments['/issues/5/comments'] = [{'id': 123}]
        self.assertEqual(p.publish(self.api, self.values, self.environ), self.pr['html_url'])
        self.assertTrue(all(call[0] == 'GET' for call in self.calls))
        self.assertFalse(any(call[1] == p.ROOT + '/pulls/5' for call in self.calls))

    def test_existing_replacement_changed_sha_denied(self):
        self.existing = [copy.deepcopy(self.pr)]
        self.existing[0]['base']['sha'] = 'b' * 40
        with self.assertRaises(p.Failure):
            p.publish(self.api, self.values, self.environ)
        self.assertTrue(all(call[0] == 'GET' for call in self.calls))

    def test_comment_race_after_create_reports_uncertain_without_retry(self):
        self.after_post = lambda: self.comments['/pulls/5/reviews'].append({'id': 1})
        with self.assertRaisesRegex(p.Failure, 'outcome uncertain'):
            p.publish(self.api, self.values, self.environ)
        self.assertEqual(sum(call[0] == 'POST' for call in self.calls), 1)

    def test_normal_heads_still_target_master(self):
        self.assertEqual(p.target_base(dict(self.values, head='codex/ordinary')), 'master')
        self.assertEqual(p.target_base(dict(self.values, head='codex/decisions-api')), 'master')


if __name__ == '__main__':
    unittest.main()
