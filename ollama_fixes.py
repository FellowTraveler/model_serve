"""Re-apply repo-stored Modelfile fixes to Ollama models with broken upstream settings.

Some models pulled from Hugging Face come with a broken chat template or stop list,
so `ollama run` returns empty replies. A fix file in OLLAMA_FIXES_DIR rebuilds the
model under its own name (`FROM <model name>` plus overrides). A re-pull restores
the broken settings, so ./model re-applies the fix after every pull and sync.
"""

import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

FIX_FILE_SUFFIX = '.Modelfile'
SHOW_TIMEOUT_SECONDS = 10
# Every Harmony-style reply opens with these, so stopping on one ends it at once.
REPLY_HEADER_TOKENS = ('<|start|>', '<|channel|>', '<|message|>')
# '<|start|>' with its '<|s' missing, as seen in broken HF templates. Any 'tart|>'
# not preceded by 's' (so '<|start|>' and '<|im_start|>' are fine).
TRUNCATED_START_TOKEN = re.compile(r'(?<!s)tart\|>')
STOP_PARAMETER = re.compile(r'^stop\s+"?(.*?)"?\s*$')


class OllamaModelFixes:
    """Finds models with known-broken settings and re-applies their fix files."""

    def __init__(self, fixes_dir: Path, ollama_base: str, run=subprocess.run):
        self.fixes_dir = Path(fixes_dir)
        self.ollama_base = ollama_base.rstrip('/')
        self.run = run

    def fix_files(self) -> dict[str, Path]:
        """Map each fixed model name (the FROM line) to its fix file."""
        if not self.fixes_dir.is_dir():
            return {}
        found = {}
        for path in sorted(self.fixes_dir.glob('*' + FIX_FILE_SUFFIX)):
            for line in path.read_text().splitlines():
                if line.upper().startswith('FROM '):
                    found[line[5:].strip()] = path
                    break
        return found

    def apply(self, model_name: str) -> bool:
        """Rebuild model_name from its fix file. False if it has none or create fails."""
        path = self.fix_files().get(model_name)
        if path is None:
            return False
        result = self.run(['ollama', 'create', model_name, '-f', str(path)],
                          capture_output=True, text=True, check=False)
        return result.returncode == 0

    def show(self, model_name: str) -> dict:
        """Ollama's /api/show for a model (its parameters text and template)."""
        request = urllib.request.Request(
            f'{self.ollama_base}/api/show',
            data=json.dumps({'model': model_name}).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(request, timeout=SHOW_TIMEOUT_SECONDS) as response:
            return json.loads(response.read())

    def problems(self, model_name: str) -> list[str]:
        """Known breakages in a model's current settings (empty if healthy)."""
        info = self.show(model_name)
        found = []
        for line in info.get('parameters', '').splitlines():
            match = STOP_PARAMETER.match(line.strip())
            if match and match.group(1) in REPLY_HEADER_TOKENS:
                found.append(f'stops on {match.group(1)}')
        if TRUNCATED_START_TOKEN.search(info.get('template', '')):
            found.append('template has truncated <|start|> token')
        return found

    def repair_broken(self, model_names: list[str]) -> tuple[list[str], dict[str, list[str]]]:
        """Re-apply fixes to broken models.

        Returns (models repaired, {broken model with no fix file: its problems}).
        """
        fixable = self.fix_files()
        repaired, unfixed = [], {}
        for model_name in model_names:
            try:
                found = self.problems(model_name)
            except urllib.error.HTTPError:
                continue  # Ollama answered but can't show this model (e.g. removed)
            except OSError:
                return [], {}  # Ollama not reachable: nothing to check against
            if not found:
                continue
            if model_name in fixable and self.apply(model_name):
                repaired.append(model_name)
            else:
                unfixed[model_name] = found
        return repaired, unfixed
