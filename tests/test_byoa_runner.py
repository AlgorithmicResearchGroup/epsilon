import sys

from runtime import byoa_runner


def test_read_adapter_command_uses_entrypoint(monkeypatch):
    monkeypatch.setenv("COLLAB_AGENT_ADAPTER_ENTRY", "examples/byoa/simple_run_agent.py:run")
    monkeypatch.setenv("COLLAB_AGENT_ADAPTER_CMD", "python3 ignored.py")
    cmd, source = byoa_runner._read_adapter_command()
    assert cmd == [sys.executable, "-m", "runtime.byoa_function_runner"]
    assert source.startswith("function:")


def test_read_adapter_command_falls_back_to_command(monkeypatch):
    monkeypatch.delenv("COLLAB_AGENT_ADAPTER_ENTRY", raising=False)
    monkeypatch.setenv("COLLAB_AGENT_ADAPTER_CMD", "python3 examples/byoa/while_loop_agent.py")
    cmd, source = byoa_runner._read_adapter_command()
    assert cmd == ["python3", "examples/byoa/while_loop_agent.py"]
    assert source == "command"
