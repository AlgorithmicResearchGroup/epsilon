from agent.tools.bash import bash_tool


def test_run_bash_normalizes_millisecond_timeout(monkeypatch):
    captured = {}

    def _fake_run(self, command, cwd=None):
        captured["timeout"] = self.timeout
        return {
            "tool": "run_bash",
            "status": "success",
            "returncode": 0,
            "attempt": command,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(bash_tool.BashRunnerActor, "run", _fake_run)

    bash_tool.run_bash({"script": "echo ok", "timeout": 120000})
    assert captured["timeout"] == 120


def test_run_bash_keeps_seconds_timeout(monkeypatch):
    captured = {}

    def _fake_run(self, command, cwd=None):
        captured["timeout"] = self.timeout
        return {
            "tool": "run_bash",
            "status": "success",
            "returncode": 0,
            "attempt": command,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(bash_tool.BashRunnerActor, "run", _fake_run)

    bash_tool.run_bash({"script": "echo ok", "timeout": 95})
    assert captured["timeout"] == 95


def test_is_suppressed_stderr_line_for_litellm_info():
    noisy = "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()`."
    assert bash_tool._is_suppressed_stderr_line(noisy) is True
    assert bash_tool._is_suppressed_stderr_line("real stderr failure line") is False
