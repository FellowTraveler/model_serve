"""Tests for ./model restarting itself under the project's Python (the uv venv)."""

import importlib.machinery
import importlib.util
from pathlib import Path
from unittest.mock import patch

MODEL_SCRIPT = Path(__file__).resolve().parent.parent / 'model'


def load_model_cli():
    """Import the extensionless ./model script as a module."""
    loader = importlib.machinery.SourceFileLoader('model_cli_reexec', str(MODEL_SCRIPT))
    spec = importlib.util.spec_from_loader('model_cli_reexec', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


model_cli = load_model_cli()


def make_interpreter(path):
    path.write_text('#!/bin/sh\n')
    path.chmod(0o755)
    return path


class TestReexecInProjectPython:
    def test_other_interpreter_reexecs_into_project_python(self, tmp_path, monkeypatch):
        venv_python = make_interpreter(tmp_path / 'python')
        monkeypatch.setenv('PYTHON_BIN', str(venv_python))
        monkeypatch.delenv(model_cli.REEXEC_MARKER, raising=False)
        monkeypatch.setattr(model_cli.sys, 'executable', '/opt/homebrew/bin/python3')
        monkeypatch.setattr(model_cli.sys, 'argv', ['./model', 'status'])
        with patch.object(model_cli.os, 'execv') as execv:
            model_cli.reexec_in_project_python()
        execv.assert_called_once_with(
            str(venv_python), [str(venv_python), str(MODEL_SCRIPT), 'status'])
        assert model_cli.os.environ[model_cli.REEXEC_MARKER] == '1'

    def test_defaults_to_uv_venv(self, monkeypatch):
        monkeypatch.delenv('PYTHON_BIN', raising=False)
        assert model_cli.project_python() == str(MODEL_SCRIPT.parent / '.venv' / 'bin' / 'python')

    def test_already_project_python_does_nothing(self, tmp_path, monkeypatch):
        venv_python = make_interpreter(tmp_path / 'python')
        monkeypatch.setenv('PYTHON_BIN', str(venv_python))
        monkeypatch.delenv(model_cli.REEXEC_MARKER, raising=False)
        monkeypatch.setattr(model_cli.sys, 'executable', str(venv_python))
        with patch.object(model_cli.os, 'execv') as execv:
            model_cli.reexec_in_project_python()
        execv.assert_not_called()

    def test_marker_prevents_loop_and_is_cleared_for_children(self, tmp_path, monkeypatch):
        venv_python = make_interpreter(tmp_path / 'python')
        monkeypatch.setenv('PYTHON_BIN', str(venv_python))
        monkeypatch.setenv(model_cli.REEXEC_MARKER, '1')
        monkeypatch.setattr(model_cli.sys, 'executable', '/opt/homebrew/bin/python3')
        with patch.object(model_cli.os, 'execv') as execv:
            model_cli.reexec_in_project_python()
        execv.assert_not_called()
        assert model_cli.REEXEC_MARKER not in model_cli.os.environ

    def test_missing_venv_keeps_current_interpreter(self, tmp_path, monkeypatch):
        monkeypatch.setenv('PYTHON_BIN', str(tmp_path / 'absent' / 'python'))
        monkeypatch.delenv(model_cli.REEXEC_MARKER, raising=False)
        with patch.object(model_cli.os, 'execv') as execv:
            model_cli.reexec_in_project_python()
        execv.assert_not_called()
