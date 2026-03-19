from datetime import datetime

from agent.memory import AgentMemory
from agent.tool_registry import Tool, get_all_tools
from agent.prompts import get_worker_system_prompt, get_initial_prompt
from agent.models.litellm_model import LiteLLMModel

MAX_TOOL_OUTPUT_CHARS = 10000

COST_PER_MILLION = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-5.2": {"input": 2.50, "output": 10.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.2": {"input": 2.50, "output": 10.00},
    "anthropic/claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "anthropic/claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}


class Worker:
    def __init__(
        self,
        user_id,
        run_id,
        user_query,
        worker_number,
        model_name,
        max_tokens,
        work_dir,
        tool_class=None,
        agents_md="",
        protocol_enabled=False,
        protocol_config=None,
        message_buffer=None,
        message_lock=None,
        shared_workspace="",
        work_queue_enabled=False,
    ):
        self.user_id = user_id
        self.run_id = run_id
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.user_query = user_query
        self.worker_number = worker_number
        self.work_dir = work_dir
        self.task_number = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.total_cost = 0.0
        self.start_time = datetime.now()

        self.protocol_enabled = protocol_enabled
        self.message_buffer = message_buffer
        self.message_lock = message_lock

        self.system_prompt = get_worker_system_prompt(self.work_dir, agents_md=agents_md, protocol_config=protocol_config, shared_workspace=shared_workspace)
        self.memory = AgentMemory()
        self.ToolClass = tool_class if tool_class is not None else Tool

        # Create model ONCE
        tools = get_all_tools(protocol_enabled, work_queue_enabled)
        self.model = LiteLLMModel(self.system_prompt, tools, model_name=self.model_name, max_tokens=self.max_tokens)

        self.initialized = False
        self.has_pending_tool_result = False
        self.last_tool_output = ""

        self.consecutive_check_messages = 0

        # Planning state
        self.planning_mode = False
        self.plan_goal = ""
        self.plan = []                # list of subtask strings
        self.current_subtask = 0      # index into self.plan
        self.completed_subtasks = []  # list of (description, summary) tuples

    def run_step(self, elapsed_time):
        """Run one step in the multi-turn conversation. Returns result dict."""
        if not self.initialized:
            prompt = get_initial_prompt(self.user_query)
            tool_name, response_data, total_tokens, prompt_tokens, response_tokens = (
                self.model.initial_request(prompt)
            )
            self.initialized = True
        elif self.has_pending_tool_result:
            tool_name, response_data, total_tokens, prompt_tokens, response_tokens = (
                self.model.send_tool_result(self.last_tool_output)
            )
        else:
            # Model returned text last time, nudge it to use a tool
            tool_name, response_data, total_tokens, prompt_tokens, response_tokens = (
                self.model.send_user_message("Please call a tool to continue working on the task.")
            )

        self.total_prompt_tokens += prompt_tokens
        self.total_response_tokens += response_tokens
        step_cost = self._estimate_cost(prompt_tokens, response_tokens)
        self.total_cost += step_cost

        print(f"Tool: {tool_name} | Tokens: {prompt_tokens:,} in / {response_tokens:,} out | ${step_cost:.4f}")
        print(f"  Cumulative: {self.total_prompt_tokens + self.total_response_tokens:,} tokens | ${self.total_cost:.4f}")

        if tool_name is None:
            # LLM returned text instead of a tool call
            print(f"LLM returned text: {response_data}")
            self.has_pending_tool_result = False
            return {
                "subtask_result": {
                    "tool": "none",
                    "status": "no_tool_call",
                    "attempt": "LLM returned text instead of tool call",
                    "stdout": str(response_data),
                    "stderr": "",
                },
                "attempted": "no",
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
            }

        # Intercept plan/submit_plan/mark_complete before dispatch
        if tool_name == "plan":
            tool_output = self._handle_plan(response_data)
        elif tool_name == "submit_plan":
            tool_output = self._handle_submit_plan(response_data)
        elif tool_name == "mark_complete":
            tool_output = self._handle_mark_complete(response_data)
        else:
            task = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "parameters": response_data,
                },
            }
            tool_output = self.ToolClass(task, self.work_dir).run()

        # Format and store tool output for next API call
        formatted = self._format_tool_output(tool_output)
        if tool_name not in ("plan", "submit_plan", "mark_complete"):
            formatted = self._augment_with_plan_context(formatted)
        if self.protocol_enabled:
            formatted = self._augment_with_message_notification(formatted)

        # Detect check_messages spin loops
        if tool_name == "check_messages":
            self.consecutive_check_messages += 1
            if self.consecutive_check_messages >= 3:
                formatted += (
                    f"\n\n[WARNING: You have called check_messages {self.consecutive_check_messages} times in a row with no results. "
                    "Stop waiting and start working on your task. You can check messages again later after making progress. "
                    "Look in the shared/ directory to see if files you need already exist.]"
                )
        else:
            self.consecutive_check_messages = 0

        self.last_tool_output = formatted
        self.has_pending_tool_result = True

        # Log to SQLite
        if isinstance(tool_output, dict):
            self.memory.save_conversation_memory(
                user_id=self.user_id,
                run_id=self.run_id,
                previous_subtask_tool=tool_output.get("tool", tool_name),
                previous_subtask_result=tool_output.get("status", "unknown"),
                previous_subtask_attempt=str(tool_output.get("attempt", "")),
                previous_subtask_output=str(tool_output.get("stdout", "")),
                previous_subtask_errors=str(tool_output.get("stderr", "")),
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )

        self.task_number += 1

        return {
            "subtask_result": tool_output,
            "attempted": "yes",
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
        }

    def _handle_plan(self, params):
        self.planning_mode = True
        self.plan_goal = params.get("goal", "")
        self.plan = []
        self.current_subtask = 0
        self.completed_subtasks = []

        print("--- PLANNING MODE ---")
        if self.plan_goal:
            print(f"Goal: {self.plan_goal}")
        print("Explore the codebase, then call submit_plan with your subtasks.")

        return {
            "tool": "plan",
            "status": "planning_mode",
            "attempt": "Entered planning mode",
            "stdout": "Planning mode active. Explore the codebase, then call submit_plan.",
            "stderr": "",
        }

    def _handle_submit_plan(self, params):
        self.planning_mode = False
        self.plan = params["subtasks"]
        self.current_subtask = 0
        self.completed_subtasks = []

        n = len(self.plan)
        print(f"--- PLAN ({n} subtasks) ---")
        for i, s in enumerate(self.plan):
            print(f"  {i+1}. {s}")

        return {
            "tool": "submit_plan",
            "status": "plan_created",
            "attempt": f"Plan created with {n} subtasks",
            "stdout": f"Plan created. Now working on subtask 1/{n}: {self.plan[0]}",
            "stderr": "",
        }

    def _handle_mark_complete(self, params):
        description = self.plan[self.current_subtask]
        summary = params["summary"]
        self.completed_subtasks.append((description, summary))
        self.current_subtask += 1

        n = len(self.plan)
        print(f"  ✓ Subtask {self.current_subtask}/{n} complete: {description}")
        print(f"    Summary: {summary}")

        if self.current_subtask < n:
            next_desc = self.plan[self.current_subtask]
            return {
                "tool": "mark_complete",
                "status": "subtask_complete",
                "attempt": f"Completed subtask {self.current_subtask}/{n}",
                "stdout": f"Subtask {self.current_subtask}/{n} complete. Now working on subtask {self.current_subtask + 1}/{n}: {next_desc}",
                "stderr": "",
            }
        else:
            return {
                "tool": "mark_complete",
                "status": "all_complete",
                "attempt": f"All {n} subtasks complete",
                "stdout": f"All {n} subtasks complete. Call done.",
                "stderr": "",
            }

    def _estimate_cost(self, prompt_tokens, response_tokens):
        rates = COST_PER_MILLION.get(self.model_name)
        if not rates and "/" in self.model_name:
            rates = COST_PER_MILLION.get(self.model_name.split("/", 1)[1])
        if not rates:
            rates = COST_PER_MILLION.get(self.model_name.rsplit("/", 1)[-1])
        if not rates:
            return 0.0
        return (prompt_tokens * rates["input"] + response_tokens * rates["output"]) / 1_000_000

    def _augment_with_plan_context(self, formatted_output):
        if self.planning_mode:
            lines = ["", "[PLANNING MODE — Explore the codebase, then call submit_plan with your subtasks]"]
            if self.plan_goal:
                lines.append(f"[Goal: {self.plan_goal}]")
            return formatted_output + "\n".join(lines)

        if not self.plan or self.current_subtask >= len(self.plan):
            return formatted_output

        n = len(self.plan)
        lines = [
            "",
            f"[PLAN PROGRESS: Subtask {self.current_subtask + 1}/{n}: {self.plan[self.current_subtask]}]",
        ]
        if self.completed_subtasks:
            lines.append("Completed:")
            for desc, summary in self.completed_subtasks:
                lines.append(f"  ✓ {desc} — {summary}")
        remaining = self.plan[self.current_subtask + 1:]
        if remaining:
            lines.append("Remaining:")
            for i, desc in enumerate(remaining):
                lines.append(f"  {self.current_subtask + 2 + i}. {desc}")

        return formatted_output + "\n".join(lines)

    def _augment_with_message_notification(self, formatted_output):
        with self.message_lock:
            count = len(self.message_buffer)
        if count > 0:
            formatted_output += f"\n\n[You have {count} new message(s) from other agents. Call check_messages to read them.]"
        return formatted_output

    def _format_tool_output(self, tool_output):
        """Convert tool output dict to a string for the API tool result message."""
        if not isinstance(tool_output, dict):
            return str(tool_output)

        parts = []
        stdout = tool_output.get("stdout", "")
        stderr = tool_output.get("stderr", "")
        if stdout:
            parts.append(str(stdout))
        if stderr:
            parts.append(f"STDERR: {stderr}")
        if not parts:
            status = tool_output.get("status", "")
            attempt = tool_output.get("attempt", "")
            parts.append(f"{attempt} - {status}" if attempt else "Done.")

        result = "\n".join(parts)
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            half = MAX_TOOL_OUTPUT_CHARS // 2
            result = (
                result[:half]
                + f"\n\n... ({len(result)} chars total, {len(result) - MAX_TOOL_OUTPUT_CHARS} chars dropped from middle. Write results to a file instead of printing them.) ...\n\n"
                + result[-half:]
            )
        return result
