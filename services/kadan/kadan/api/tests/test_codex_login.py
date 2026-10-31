import importlib.util
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).parents[3] / "sandbox" / "codex_login.py"
spec = importlib.util.spec_from_file_location("codex_login", path)
login = importlib.util.module_from_spec(spec)
spec.loader.exec_module(login)


def test_login_returns_existing_device_prompt():
    prompt = {"state": "signing_in", "message": "Enter the code", "login": {"url": "https://example.com", "code": "ABCD-EFGH"}}
    with patch.object(login, "setup", return_value=prompt), patch.object(login, "launch") as launch:
        assert login.login() == prompt
        launch.assert_not_called()


def test_login_starts_device_authorization():
    with patch.object(login, "setup", return_value={"state": "signed_out"}), patch.object(login, "launch", return_value={"pid": 3}), patch.object(login, "write_json") as write, patch.object(login, "wait_for_login", return_value={"state": "signing_in"}):
        assert login.login() == {"state": "signing_in"}
        assert write.call_args.args[1] == {"pid": 3}


def test_prompt_extracts_link_and_code(tmp_path):
    (tmp_path / "login.log").write_text("Open https://example.com/device and enter ABCD-EFGH")
    with patch.object(login, "STATE", tmp_path):
        assert login.login_prompt() == {"url": "https://example.com/device", "code": "ABCD-EFGH"}
