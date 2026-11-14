"""Tests for re-applying repo-stored Modelfile fixes to Ollama models."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from ollama_fixes import OllamaModelFixes

REPO_FIXES_DIR = Path(__file__).resolve().parent.parent / 'ollama_fixes'
BROKEN = 'hf.co/someone/broken-GGUF:Q8_0'
HEALTHY = 'qwen3:8b-q8_0'


def write_fix(fixes_dir, model_name, body='PARAMETER stop <|eot|>\n'):
    fixes_dir.mkdir(exist_ok=True)
    path = fixes_dir / (model_name.replace('/', '_').replace(':', '_') + '.Modelfile')
    path.write_text(f'# why this fix exists\nFROM {model_name}\n{body}')
    return path


def fake_show(models):
    """Fake Ollama /api/show: models maps name -> (parameters text, template)."""
    def show(name):
        parameters, template = models[name]
        return {'parameters': parameters, 'template': template}
    return show


def make_fixes(tmp_path, models=None):
    run = MagicMock(return_value=MagicMock(returncode=0))
    fixes = OllamaModelFixes(tmp_path / 'fixes', 'http://ollama.invalid', run=run)
    fixes.show = fake_show(models or {})
    return fixes, run


class TestFixFiles:
    def test_maps_model_named_in_from_line_to_its_file(self, tmp_path):
        fixes, _ = make_fixes(tmp_path)
        path = write_fix(tmp_path / 'fixes', BROKEN)
        assert fixes.fix_files() == {BROKEN: path}

    def test_missing_dir_means_no_fixes(self, tmp_path):
        fixes, _ = make_fixes(tmp_path)
        assert fixes.fix_files() == {}

    def test_repo_fix_files_each_name_a_model(self):
        """Every shipped fix file must FROM a model name, not a blob path."""
        found = OllamaModelFixes(REPO_FIXES_DIR, 'http://ollama.invalid').fix_files()
        assert found, 'expected shipped fix files'
        for model_name in found:
            assert not model_name.startswith('/'), model_name


class TestApply:
    def test_runs_ollama_create_with_fix_file(self, tmp_path):
        fixes, run = make_fixes(tmp_path)
        path = write_fix(tmp_path / 'fixes', BROKEN)
        assert fixes.apply(BROKEN) is True
        run.assert_called_once_with(['ollama', 'create', BROKEN, '-f', str(path)],
                                    capture_output=True, text=True, check=False)

    def test_model_without_fix_is_left_alone(self, tmp_path):
        fixes, run = make_fixes(tmp_path)
        assert fixes.apply(HEALTHY) is False
        run.assert_not_called()


class TestProblems:
    def test_stop_on_reply_header_token_is_reported(self, tmp_path):
        fixes, _ = make_fixes(tmp_path, {BROKEN: ('stop "<|channel|>"\nstop "<|end|>"', '')})
        assert fixes.problems(BROKEN) == ['stops on <|channel|>']

    def test_truncated_start_token_is_reported(self, tmp_path):
        fixes, _ = make_fixes(tmp_path, {BROKEN: ('', '{{ if .Prompt }}tart|>user<|message|>')})
        assert fixes.problems(BROKEN) == ['template has truncated <|start|> token']

    def test_healthy_model_has_no_problems(self, tmp_path):
        template = '<|start|>user<|message|>{{ .Prompt }}<|end|>'
        fixes, _ = make_fixes(tmp_path, {HEALTHY: ('stop "<|return|>"\ntemperature 1', template)})
        assert fixes.problems(HEALTHY) == []

    def test_im_start_token_is_not_mistaken_for_truncation(self, tmp_path):
        template = '<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant'
        fixes, _ = make_fixes(tmp_path, {HEALTHY: ('', template)})
        assert fixes.problems(HEALTHY) == []

    def test_longer_stop_containing_header_token_is_fine(self, tmp_path):
        fixes, _ = make_fixes(tmp_path, {HEALTHY: ('stop "<|start|>user<|message|>"', '')})
        assert fixes.problems(HEALTHY) == []


class TestRepairBroken:
    def test_reapplies_fix_and_reports_unfixable(self, tmp_path):
        other = 'hf.co/someone/other-GGUF:Q4_K_M'
        fixes, run = make_fixes(tmp_path, {
            BROKEN: ('stop "<|start|>"', ''),
            other: ('stop "<|message|>"', ''),
            HEALTHY: ('', ''),
        })
        write_fix(tmp_path / 'fixes', BROKEN)
        repaired, unfixed = fixes.repair_broken([BROKEN, other, HEALTHY])
        assert repaired == [BROKEN]
        assert unfixed == {other: ['stops on <|message|>']}
        assert run.call_count == 1
        assert run.call_args.args[0][:3] == ['ollama', 'create', BROKEN]

    def test_unreachable_ollama_skips_check(self, tmp_path):
        fixes, run = make_fixes(tmp_path)
        fixes.show = MagicMock(side_effect=OSError('connection refused'))
        assert fixes.repair_broken([BROKEN]) == ([], {})
        run.assert_not_called()


    def test_model_ollama_cannot_show_is_skipped_not_fatal(self, tmp_path):
        import urllib.error
        fixes, run = make_fixes(tmp_path, {BROKEN: ('stop "<|start|>"', '')})
        write_fix(tmp_path / 'fixes', BROKEN)
        real_show = fixes.show

        def show(name):
            if name == 'gone:latest':
                raise urllib.error.HTTPError('u', 404, 'not found', None, None)
            return real_show(name)
        fixes.show = show
        assert fixes.repair_broken(['gone:latest', BROKEN]) == ([BROKEN], {})
        assert run.call_count == 1


    def test_failed_create_is_reported_as_unfixed(self, tmp_path):
        fixes, run = make_fixes(tmp_path, {BROKEN: ('stop "<|start|>"', '')})
        write_fix(tmp_path / 'fixes', BROKEN)
        run.return_value = MagicMock(returncode=1)
        assert fixes.repair_broken([BROKEN]) == ([], {BROKEN: ['stops on <|start|>']})


class TestShow:
    def test_posts_model_name_to_api_show(self, tmp_path, monkeypatch):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({'parameters': 'temperature 1', 'template': 'T'}).encode()

        def urlopen(request, timeout):
            captured['url'] = request.full_url
            captured['body'] = json.loads(request.data)
            return Response()

        import ollama_fixes
        monkeypatch.setattr(ollama_fixes.urllib.request, 'urlopen', urlopen)
        fixes = OllamaModelFixes(tmp_path, 'http://ollama.invalid')
        assert fixes.show(BROKEN) == {'parameters': 'temperature 1', 'template': 'T'}
        assert captured == {'url': 'http://ollama.invalid/api/show', 'body': {'model': BROKEN}}
