"""Tests for the model CLI's automatic log trimming."""

import importlib.machinery
import importlib.util
from pathlib import Path

MODEL_SCRIPT = Path(__file__).resolve().parent.parent / 'model'


def load_model_cli():
    """Import the extensionless ./model script as a module."""
    loader = importlib.machinery.SourceFileLoader('model_cli_trim', str(MODEL_SCRIPT))
    spec = importlib.util.spec_from_loader('model_cli_trim', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


model_cli = load_model_cli()


def write_lines(path, count):
    path.write_text(''.join(f'line {i:06d} ' + 'x' * 80 + '\n' for i in range(count)))


class TestTrimLogFile:
    def test_oversized_log_keeps_newest_half_on_line_boundary(self, tmp_path):
        log = tmp_path / 'serve.log'
        write_lines(log, 5000)  # ~445 KB
        max_bytes = 100_000

        model_cli.trim_log_file(log, max_bytes)

        text = log.read_text()
        assert len(text.encode()) <= max_bytes // 2
        assert text.startswith('line ')            # starts on a whole line
        assert text.endswith('line 004999 ' + 'x' * 80 + '\n')  # newest line kept

    def test_trim_keeps_same_file_for_open_appenders(self, tmp_path):
        log = tmp_path / 'serve.log'
        write_lines(log, 5000)
        inode_before = log.stat().st_ino
        with open(log, 'a') as writer:
            model_cli.trim_log_file(log, 100_000)
            writer.write('after trim\n')
        assert log.stat().st_ino == inode_before
        assert log.read_text().endswith('after trim\n')

    def test_small_log_untouched(self, tmp_path):
        log = tmp_path / 'serve.log'
        write_lines(log, 10)
        before = log.read_bytes()
        model_cli.trim_log_file(log, 100_000)
        assert log.read_bytes() == before

    def test_missing_log_is_fine(self, tmp_path):
        model_cli.trim_log_file(tmp_path / 'absent.log', 100_000)

    def test_zero_limit_disables_trimming(self, tmp_path):
        log = tmp_path / 'serve.log'
        write_lines(log, 5000)
        before = log.read_bytes()
        model_cli.trim_log_file(log, 0)
        assert log.read_bytes() == before


class TestLogLimitFromEnv:
    def test_default_limit(self, monkeypatch):
        monkeypatch.delenv('MODEL_SERVE_LOG_MAX_MB', raising=False)
        assert model_cli.log_max_bytes() == 50 * 1024 * 1024

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv('MODEL_SERVE_LOG_MAX_MB', '5')
        assert model_cli.log_max_bytes() == 5 * 1024 * 1024


class TestReloadLlamaSwapLogging:
    def test_restarted_llama_swap_logs_to_log_file(self, tmp_path, monkeypatch):
        """A llama-swap restarted by ./model sync must append to the log, not /dev/null."""
        import time
        from unittest.mock import patch

        log = tmp_path / 'serve.log'
        monkeypatch.setattr(model_cli, 'LOG_FILE', log)
        pgrep = type('Result', (), {'returncode': 0, 'stdout': '12345\n'})()
        with patch.object(model_cli.subprocess, 'run', return_value=pgrep), \
             patch.object(model_cli.os, 'kill'), \
             patch.object(time, 'sleep'), \
             patch.object(model_cli.subprocess, 'Popen') as popen:
            assert model_cli.reload_llama_swap() is True

        stdout = popen.call_args.kwargs['stdout']
        assert stdout.name == str(log)
        assert 'a' in stdout.mode
        assert popen.call_args.kwargs['stderr'] == model_cli.subprocess.STDOUT
