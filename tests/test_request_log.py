"""Tests for request logging: metadata by default, private files, retention."""
import json
import os
import stat
import time

from request_log import RequestLogger, env_flag

BODY = {
    "model": "ms/gemma4-26b-a4b-it-q4_k_m",
    "stream": True,
    "messages": [{"role": "user", "content": "PRIVATE EMAIL TEXT"}],
    "tools": [{"type": "function", "function": {"name": "search_email"}}],
}


def test_default_log_has_metadata_but_no_content(tmp_path):
    path = RequestLogger(tmp_path, log_bodies=False, retention_days=7).log_request(
        "/v1/chat/completions", BODY, "rustymail")
    text = path.read_text()
    assert "PRIVATE EMAIL TEXT" not in text
    record = json.loads(text)
    assert record["model"] == "ms/gemma4-26b-a4b-it-q4_k_m"
    assert record["message_count"] == 1
    assert record["tool_count"] == 1
    assert record["source"] == "rustymail"
    assert "body" not in record
    # file name keeps the model, so per-model usage can still be counted from names
    assert "_ms_gemma4-26b-a4b-it-q4_k_m__v1_chat_completions.json" in path.name


def test_bodies_are_logged_only_when_enabled(tmp_path):
    path = RequestLogger(tmp_path, log_bodies=True, retention_days=7).log_request("/v1/chat/completions", BODY)
    assert json.loads(path.read_text())["body"] == BODY


def test_no_streamed_response_log_unless_bodies_enabled(tmp_path):
    assert RequestLogger(tmp_path, log_bodies=False, retention_days=7).response_log_path("ms/a") is None
    path = RequestLogger(tmp_path, log_bodies=True, retention_days=7).response_log_path("ms/a")
    assert path.exists()


def test_folder_and_files_are_private(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(mode=0o755)
    request_logger = RequestLogger(log_dir, log_bodies=True, retention_days=7)
    path = request_logger.log_request("/v1/chat/completions", BODY)
    response = request_logger.response_log_path("ms/a")
    assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(response.stat().st_mode) == 0o600


def test_prune_deletes_only_files_past_retention(tmp_path):
    request_logger = RequestLogger(tmp_path, log_bodies=False, retention_days=7)
    old = request_logger.log_request("/v1/chat/completions", BODY)
    new = request_logger.log_request("/v1/chat/completions", BODY)
    eight_days_ago = time.time() - 8 * 86400
    os.utime(old, (eight_days_ago, eight_days_ago))
    assert request_logger.prune() == 1
    assert not old.exists()
    assert new.exists()


def test_env_flag():
    assert env_flag("true") and env_flag("1") and env_flag("YES")
    assert not env_flag("false") and not env_flag("") and not env_flag(None)
