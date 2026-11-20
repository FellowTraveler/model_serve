"""Tests for ./model start/stop when the launchd agent (setup_service.sh) is installed."""

import importlib.machinery
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

MODEL_SCRIPT = Path(__file__).resolve().parent.parent / 'model'


def load_model_cli():
    """Import the extensionless ./model script as a module."""
    loader = importlib.machinery.SourceFileLoader('model_cli_service', str(MODEL_SCRIPT))
    spec = importlib.util.spec_from_loader('model_cli_service', loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


model_cli = load_model_cli()


def launchctl_calls(run_mock):
    return [call.args[0] for call in run_mock.call_args_list if call.args[0][0] == 'launchctl']


def fake_run(loaded: bool):
    """subprocess.run stand-in: `launchctl print` succeeds only when the agent is loaded."""
    def run(argv, **kwargs):
        result = MagicMock()
        result.returncode = 0 if (argv[:2] != ['launchctl', 'print'] or loaded) else 113
        return result
    return MagicMock(side_effect=run)


class TestServiceAwareStartStop:
    def install_agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_cli.sys, 'platform', 'darwin')
        monkeypatch.setattr(model_cli.Path, 'home', lambda: tmp_path)
        monkeypatch.setenv('MODEL_SERVE_SERVICE_LABEL', 'test.agent')
        plist = tmp_path / 'Library' / 'LaunchAgents' / 'test.agent.plist'
        plist.parent.mkdir(parents=True)
        plist.write_text('<plist/>')
        return plist

    def test_start_with_unloaded_agent_bootstraps_it_and_never_runs_start_sh(self, tmp_path, monkeypatch):
        plist = self.install_agent(tmp_path, monkeypatch)
        run = fake_run(loaded=False)
        with patch.object(model_cli.subprocess, 'run', run), \
             patch.object(model_cli.subprocess, 'Popen') as popen:
            model_cli.cmd_start([])
        popen.assert_not_called()
        uid = model_cli.os.getuid()
        assert ['launchctl', 'bootstrap', f'gui/{uid}', str(plist)] in launchctl_calls(run)

    def test_start_with_loaded_agent_kickstarts_it(self, tmp_path, monkeypatch):
        self.install_agent(tmp_path, monkeypatch)
        run = fake_run(loaded=True)
        with patch.object(model_cli.subprocess, 'run', run), \
             patch.object(model_cli.subprocess, 'Popen') as popen:
            model_cli.cmd_start([])
        popen.assert_not_called()
        uid = model_cli.os.getuid()
        assert ['launchctl', 'kickstart', f'gui/{uid}/test.agent'] in launchctl_calls(run)

    def test_stop_unloads_agent_before_stop_sh(self, tmp_path, monkeypatch):
        self.install_agent(tmp_path, monkeypatch)
        run = fake_run(loaded=True)
        with patch.object(model_cli.subprocess, 'run', run):
            model_cli.cmd_stop([])
        argvs = [call.args[0] for call in run.call_args_list]
        uid = model_cli.os.getuid()
        bootout = argvs.index(['launchctl', 'bootout', f'gui/{uid}/test.agent'])
        stop_sh = argvs.index([str(model_cli.SCRIPT_DIR / 'stop.sh')])
        assert bootout < stop_sh

    def test_without_agent_start_runs_start_sh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_cli.Path, 'home', lambda: tmp_path)
        run = MagicMock(return_value=MagicMock(returncode=1))  # pgrep: nothing running
        with patch.object(model_cli.subprocess, 'run', run), \
             patch.object(model_cli.subprocess, 'Popen') as popen, \
             patch.object(model_cli, 'LOG_FILE', tmp_path / 'model_serve.log'):
            model_cli.cmd_start([])
        popen.assert_called_once()
        assert launchctl_calls(run) == []
