"""Tests for `./model list` filtering and --size sorting."""

import importlib.machinery
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

MODEL_SCRIPT = Path(__file__).resolve().parent.parent / 'model'


def load_model_cli():
    """Import the extensionless ./model script as a module."""
    loader = importlib.machinery.SourceFileLoader('model_cli', str(MODEL_SCRIPT))
    spec = importlib.util.spec_from_loader('model_cli', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


model_cli = load_model_cli()

OLLAMA_LIST_OUTPUT = (
    "NAME                    ID              SIZE      MODIFIED\n"
    "qwen3.5:9b-q4_K_M       4b53c3daca59    6.6 GB    4 weeks ago\n"
    "nomic-embed-text:latest 0a109f422b47    274 MB    2 months ago\n"
    "qwen3.6:35b-a3b-q4_K_M  b52ca43760e9    23 GB     4 weeks ago\n"
    "llama3.3:70b-q6_K       992c05c90cd8    57 GB     6 months ago\n"
)


def run_list(capsys, args):
    fake = type('Result', (), {'returncode': 0, 'stdout': OLLAMA_LIST_OUTPUT, 'stderr': ''})()
    with patch.object(model_cli.subprocess, 'run', return_value=fake):
        model_cli.cmd_list(args)
    return capsys.readouterr().out.splitlines()


def model_names(lines):
    return [line.split()[0] for line in lines[1:] if line and not line.startswith('(')]


class TestSizeParsing:
    @pytest.mark.parametrize('line,expected', [
        ("a  id  6.6 GB  4 weeks ago", 6.6e9),
        ("a  id  274 MB  2 months ago", 274e6),
        ("a  id  1.2 TB  1 day ago", 1.2e12),
        ("a  id  512 KB  now", 512e3),
    ])
    def test_parses_ollama_size_column(self, line, expected):
        assert model_cli.ollama_size_bytes(line) == pytest.approx(expected)

    def test_unparseable_line_sorts_last(self):
        assert model_cli.ollama_size_bytes("garbage line") == -1


class TestListSize:
    def test_size_sorts_largest_first_and_keeps_header(self, capsys):
        lines = run_list(capsys, ['--size'])
        assert lines[0].startswith('NAME')
        assert model_names(lines) == [
            'llama3.3:70b-q6_K', 'qwen3.6:35b-a3b-q4_K_M',
            'qwen3.5:9b-q4_K_M', 'nomic-embed-text:latest',
        ]

    def test_size_with_filter(self, capsys):
        lines = run_list(capsys, ['--size', 'qwen'])
        assert model_names(lines) == ['qwen3.6:35b-a3b-q4_K_M', 'qwen3.5:9b-q4_K_M']

    def test_filter_before_size_flag(self, capsys):
        lines = run_list(capsys, ['qwen', '--size'])
        assert model_names(lines) == ['qwen3.6:35b-a3b-q4_K_M', 'qwen3.5:9b-q4_K_M']

    def test_without_size_keeps_ollama_order(self, capsys):
        lines = run_list(capsys, ['qwen'])
        assert model_names(lines) == ['qwen3.5:9b-q4_K_M', 'qwen3.6:35b-a3b-q4_K_M']
