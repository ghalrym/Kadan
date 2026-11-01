import json
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy.dialects import postgresql

from api.orm_models.subscriptions import CapabilityProvider, Subscription
from api.repositories import subscriptions


class CredentialTests(unittest.TestCase):
    def test_codex_requires_a_chatgpt_auth_file(self):
        for credential in ("not json", json.dumps({"OPENAI_API_KEY": "sk"})):
            with self.assertRaises(subscriptions.SubscriptionError):
                subscriptions.validate_credential("codex", credential)
        auth = json.dumps({"tokens": {"access_token": "a"}})
        self.assertEqual(subscriptions.validate_credential("codex", f" {auth}\n"), auth)

    def test_claude_token_is_trimmed_and_single(self):
        self.assertEqual(
            subscriptions.validate_credential("claude", " sk-ant-oat01-x\n"),
            "sk-ant-oat01-x",
        )
        with self.assertRaises(subscriptions.SubscriptionError):
            subscriptions.validate_credential("claude", "Your token: sk-ant-oat01-x")


class ProviderTests(unittest.TestCase):
    def test_refresh_only_updates_the_matching_credential(self):
        session = MagicMock()
        with (
            patch("api.repositories.subscriptions.get_engine"),
            patch("api.repositories.subscriptions.Session") as session_class,
        ):
            session_class.return_value.__enter__.return_value = session
            subscriptions.refresh("codex", "old", "new")
        statement = session.execute.call_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        self.assertIn("subscriptions.provider =", str(compiled))
        self.assertIn("subscriptions.credential =", str(compiled).split("WHERE")[1])
        self.assertEqual(set(compiled.params.values()), {"codex", "old", "new"})
        session.commit.assert_called_once()

    def test_subscription_must_be_connected_before_it_serves_language(self):
        session = MagicMock()
        session.get.return_value = None
        with self.assertRaises(subscriptions.SubscriptionError):
            subscriptions.choose(session, "language", "claude")
        session.commit.assert_not_called()

    def test_local_needs_no_subscription(self):
        session = MagicMock()
        session.get.return_value = None
        subscriptions.choose(session, "language", "local")
        added = session.add.call_args.args[0]
        self.assertEqual((added.capability, added.provider), ("language", "local"))
        session.commit.assert_called_once()

    def test_disconnect_returns_its_capabilities_to_local(self):
        session = MagicMock()
        row = Subscription(provider="codex", credential="{}")
        served = CapabilityProvider(capability="language", provider="codex")
        vision = CapabilityProvider(capability="vision", provider="codex")
        session.get.return_value = row
        session.scalars.return_value = [served, vision]
        subscriptions.disconnect(session, "codex")
        session.delete.assert_called_once_with(row)
        self.assertEqual(served.provider, "local")
        self.assertEqual(vision.provider, "local")
        session.commit.assert_called_once()

    def test_snapshot_defaults_language_to_local(self):
        session = MagicMock()
        session.scalars.side_effect = [["claude"], []]
        result = subscriptions.snapshot(session)
        self.assertEqual(result.capabilities.language, "local")
        self.assertEqual(result.capabilities.vision, "local")
        self.assertEqual(
            [(row.provider, row.connected, row.model) for row in result.subscriptions],
            [("claude", True, "claude-haiku-5-5"), ("codex", False, "gpt-6.1-terra")],
        )

    def test_vision_selection_does_not_modify_language(self):
        session = MagicMock()
        selected = CapabilityProvider(capability="vision", provider="local")
        session.get.side_effect = [
            Subscription(provider="claude", credential="t"),
            selected,
        ]
        subscriptions.choose(session, "vision", "claude")
        self.assertEqual(selected.provider, "claude")
        session.get.assert_any_call(CapabilityProvider, "vision")
        session.commit.assert_called_once()
