import json
import unittest
from unittest.mock import patch

from api.services import subscription_login
from api.services.subscription_login import (
    CLAUDE_URL,
    CODEX_CODE,
    CODEX_URL,
    LoginFailure,
    LoginRun,
    SubscriptionLogins,
    clean,
)

CODEX_OUTPUT = (
    b"Follow these steps to sign in with ChatGPT using device code authorization:\n\n"
    b"1. Open this link in your browser and sign in to your account\n"
    b"   \x1b[94mhttps://auth.openai.com/codex/device\x1b[0m\n\n"
    b"2. Enter this one-time code \x1b[90m(expires in 15 minutes)\x1b[0m\n"
    b"   \x1b[94mTZPF-DBS7F\x1b[0m\n"
)
CLAUDE_OUTPUT = (
    b"\x1b[2GPaste\x1b[8Gcode\r\r\n"
    b"\x1b]8;id=x;https://claude.com/cai/oauth/authorize?code=true&state=abc\x07"
    b"https://claude.com/cai/oauth/authorize?code=true&state=abc\x1b]8;;\x07\r\r\n"
)


class FakeProcess:
    def __init__(self, returncode=0):
        self.returncode = None
        self.exit_code = returncode

    async def wait(self):
        self.returncode = self.exit_code
        return self.returncode

    def kill(self):
        pass


class ParsingTests(unittest.TestCase):
    def test_codex_link_and_code_survive_terminal_colors(self):
        text = clean(CODEX_OUTPUT)
        self.assertEqual(
            CODEX_URL.search(text).group(0), "https://auth.openai.com/codex/device"
        )
        self.assertEqual(CODEX_CODE.search(text).group(1), "TZPF-DBS7F")

    def test_claude_link_is_read_from_the_terminal_ui(self):
        self.assertEqual(
            CLAUDE_URL.search(clean(CLAUDE_OUTPUT)).group(0),
            "https://claude.com/cai/oauth/authorize?code=true&state=abc",
        )

    def test_claude_rejected_code_fails_instead_of_waiting(self):
        with self.assertRaises(LoginFailure):
            LoginRun._claude_token("OAuth error: Request failed  Press Enter to retry.")
        self.assertEqual(
            LoginRun._claude_token("Your token:\r\n sk-ant-oat01-Ab_c-9\r\n").group(0),
            "sk-ant-oat01-Ab_c-9",
        )


class FinishTests(unittest.IsolatedAsyncioTestCase):
    async def test_claude_token_is_saved_as_the_subscription(self):
        run = LoginRun("claude")
        run.state.status = "verifying"
        run.raw = b"Long-lived token created:\r\n\x1b[2Gsk-ant-oat01-token\r\n"
        with patch.object(subscription_login.subscriptions, "store") as store:
            await run._finish()
        store.assert_called_once_with("claude", "sk-ant-oat01-token")
        self.assertEqual(run.state.status, "complete")
        self.assertFalse(run.home.exists())

    async def test_codex_auth_file_is_saved_when_the_login_exits(self):
        run = LoginRun("codex")
        auth = json.dumps({"tokens": {"access_token": "a"}})
        (run.home / "auth.json").write_text(auth)
        run.process = FakeProcess()
        with patch.object(subscription_login.subscriptions, "store") as store:
            await run._finish()
        store.assert_called_once_with("codex", auth)
        self.assertEqual(run.state.status, "complete")

    async def test_codex_exit_without_sign_in_fails(self):
        run = LoginRun("codex")
        run.process = FakeProcess(returncode=1)
        run.raw = b"Error logging in: device code expired\n"
        with patch.object(subscription_login.subscriptions, "store") as store:
            await run._finish()
        store.assert_not_called()
        self.assertEqual(run.state.status, "failed")
        self.assertIn("device code expired", run.state.error)

    async def test_code_is_only_accepted_by_a_waiting_claude_sign_in(self):
        logins = SubscriptionLogins()
        with self.assertRaises(LoginFailure):
            await logins.submit_code("claude", "abc")
        run = logins.runs["codex"] = LoginRun("codex")
        run.state.status = "waiting"
        with self.assertRaises(LoginFailure):
            await logins.submit_code("codex", "abc")
        await logins.close()
        self.assertEqual(run.state.status, "failed")
        self.assertFalse(run.home.exists())

    async def test_missing_cli_reports_a_plain_failure(self):
        with patch(
            "api.services.subscription_login.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError,
        ):
            state = await LoginRun("codex").start()
        self.assertEqual(state.status, "failed")
        self.assertNotIn("CLI", state.error)
