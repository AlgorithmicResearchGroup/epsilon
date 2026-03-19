import logging
import os
import signal
import subprocess
import sys
import threading
from typing import Dict, Optional
from agent.utils import remove_ascii

DEFAULT_TIMEOUT = 1200
MAX_TIMEOUT_SECONDS = 86400
_LITELLM_STDERR_NOISE_MARKERS = (
    "litellm.info: if you need to debug this error, use `litellm._turn_on_debug()`",
    "litellm._turn_on_debug()",
)

bash_tool_definitions = [
    {
        "name": "run_bash",
        "description": "Run a non-interactive bash command on the server. Timeout is in seconds (not milliseconds). If a command may run for a long time (e.g. a test suite, ML training, server smoke test), set timeout accordingly. If a command times out, the process and all its children are killed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "The bash script to run.",
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Max seconds to allow this command to run before killing it. Default: {DEFAULT_TIMEOUT}. Set higher for long-running tasks (tests, builds, training). If you pass milliseconds by mistake (e.g. 120000), it will be normalized to seconds.",
                },
            },
            "required": ["script"],
        },
    },
]


def _normalize_timeout_seconds(raw_timeout) -> int:
    try:
        parsed = int(raw_timeout)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    if parsed <= 0:
        return DEFAULT_TIMEOUT
    # Common model mistake: milliseconds instead of seconds.
    if parsed >= 10000 and parsed % 1000 == 0:
        parsed = parsed // 1000
    return max(1, min(parsed, MAX_TIMEOUT_SECONDS))


def _is_suppressed_stderr_line(line: str) -> bool:
    lowered = str(line or "").strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _LITELLM_STDERR_NOISE_MARKERS)


class BashRunnerActor:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def _kill_process_group(self, process):
        """Kill the entire process group so child processes don't survive."""
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def run(self, command: str, cwd: str = None) -> Dict[str, Optional[str]]:
        """Method to execute a bash command and return the results."""
        logging.info(f"Executing command: {command}")

        result = {
            "tool": "run_bash",
            "status": "failure",
            "returncode": None,
            "attempt": command,
            "stdout": "",
            "stderr": "",
        }

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=cwd,
            start_new_session=True,
        )

        stdout_lines = []
        stderr_lines = []

        def read_stdout():
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                stdout_lines.append(line)

        def read_stderr():
            for line in iter(process.stderr.readline, ''):
                if _is_suppressed_stderr_line(line):
                    continue
                print(line, end='', file=sys.stderr)
                stderr_lines.append(line)

        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)

        stdout_thread.start()
        stderr_thread.start()

        try:
            process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            result["stdout"] = ''.join(stdout_lines)
            result["stderr"] = f"Command timed out after {self.timeout} seconds and was killed (including all child processes). If this command needs more time, call run_bash with a higher timeout value."
            logging.error(result["stderr"])
            return result

        stdout_thread.join()
        stderr_thread.join()

        returncode = process.returncode
        result['returncode'] = returncode
        result["stdout"] = ''.join(stdout_lines)
        result["stderr"] = ''.join(stderr_lines)

        if returncode == 0:
            result["status"] = "success"
        else:
            result["status"] = "failure"
            logging.error(f"Command failed with returncode: {returncode}")
            logging.error(f"Command stderr:\n{result['stderr']}")

        return result


def run_bash(arguments: dict, work_dir: str = None) -> Dict[str, Optional[str]]:
    """
    This function is used to run a bash script on the server.
    Use this function to run the code you need to complete the task.
    """
    if isinstance(arguments, dict):
        command = arguments["script"]
        timeout = _normalize_timeout_seconds(arguments.get("timeout", DEFAULT_TIMEOUT))
    else:
        command = arguments
        timeout = DEFAULT_TIMEOUT

    runner_actor = BashRunnerActor(timeout=timeout)
    result = runner_actor.run(command, cwd=work_dir)
    result["stdout"] = remove_ascii(result["stdout"])
    return result
