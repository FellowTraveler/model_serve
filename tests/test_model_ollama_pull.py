"""Tests that ./model re-applies Ollama fix files after a pull and during sync."""

import importlib.machinery
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

MODEL_SCRIPT = Path(__file__).resolve().parent.parent / 'model'


def load_model_cli():
    """Import the extensionless ./model script as a module."""
    loader = importlib.machinery.SourceFileLoader('model_cli_pull', str(MODEL_SCRIPT))
    spec = importlib.util.spec_from_loader('model_cli_pull', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


model_cli = load_model_cli()
MODEL = 'hf.co/someone/broken-GGUF:Q8_0'


def test_successful_pull_reapplies_fix_file():
    fixes = MagicMock()
    with patch.object(model_cli, 'run_cmd', return_value=MagicMock(returncode=0)), \
         patch.object(model_cli, 'ollama_fixes', return_value=fixes):
        assert model_cli.ollama_pull(MODEL) is True
    fixes.apply.assert_called_once_with(MODEL)


def test_failed_pull_does_not_touch_model():
    fixes = MagicMock()
    with patch.object(model_cli, 'run_cmd', return_value=MagicMock(returncode=1)), \
         patch.object(model_cli, 'ollama_fixes', return_value=fixes):
        assert model_cli.ollama_pull(MODEL) is False
    fixes.apply.assert_not_called()


def test_sync_warns_about_broken_model_without_fix(capsys):
    fixes = MagicMock()
    fixes.repair_broken.return_value = (['hf.co/other:Q8_0'], {MODEL: ['stops on <|channel|>']})
    with patch.object(model_cli, 'ollama_list_names', return_value=[MODEL]), \
         patch.object(model_cli, 'ollama_fixes', return_value=fixes):
        model_cli.repair_broken_ollama_models()
    fixes.repair_broken.assert_called_once_with([MODEL])
    out = capsys.readouterr().out
    assert 'Re-applied fix file for hf.co/other:Q8_0' in out
    assert f'{MODEL} looks broken' in out


def test_fixes_dir_from_env_relative_to_repo(monkeypatch):
    monkeypatch.setenv('OLLAMA_FIXES_DIR', 'custom_fixes')
    monkeypatch.setenv('OLLAMA_BASE', 'http://ollama.invalid')
    fixes = model_cli.ollama_fixes()
    assert fixes.fixes_dir == MODEL_SCRIPT.parent / 'custom_fixes'
    assert fixes.ollama_base == 'http://ollama.invalid'
