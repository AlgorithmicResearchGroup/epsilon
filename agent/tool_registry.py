from agent.tools.bash.bash_tool import run_bash, bash_tool_definitions
from agent.tools.code.code_tool import read_file, write_file, edit_file, code_tool_definitions
from agent.tools.llm.llm_tool import call_llm, llm_tool_definitions
from agent.tools.sql.sql_tool import sql_query, sql_tool_definitions
from agent.tools.web.web_tool import web_search, fetch_url, web_tool_definitions


# Protocol state — set by runtime/agent_main.py when protocol is enabled
_protocol_state = {}


messaging_tool_definitions = [
    {
        "name": "send_message",
        "description": "Send a message to other agents. If 'target' is specified, the message is sent directly to that agent only. Otherwise, it is broadcast to all subscribers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The message content to send."},
                "topic": {"type": "string", "description": "Topic/channel. Defaults to 'general'."},
                "target": {"type": "string", "description": "Optional: specific agent_id to send to directly (skips broadcast)."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "check_messages",
        "description": "Check for new messages from other agents. Returns all unread messages.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

work_queue_tool_definitions = [
    {
        "name": "submit_task",
        "description": "Submit a task to the work queue for another agent to pick up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Task description."},
                "payload": {"type": "object", "description": "Task data payload."},
            },
            "required": ["description"],
        },
    },
    {
        "name": "request_task",
        "description": "Request a task from the work queue. Returns a task assignment or 'no_tasks'.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def send_message(arguments, work_dir=None):
    agent = _protocol_state["agent"]  # KeyError if protocol not set up = fail hard
    content = arguments["content"]
    topic = arguments.get("topic", "general")
    target = arguments.get("target")
    agent.send_data(content, topic=topic, target=target)
    if target:
        return {
            "tool": "send_message",
            "status": "success",
            "attempt": f"Sent directed message to '{target}' on topic '{topic}'",
            "stdout": f"Message sent to '{target}' on '{topic}': {content[:200]}",
            "stderr": "",
        }
    return {
        "tool": "send_message",
        "status": "success",
        "attempt": f"Sent broadcast on topic '{topic}'",
        "stdout": f"Message broadcast on '{topic}': {content[:200]}",
        "stderr": "",
    }


def check_messages(arguments, work_dir=None):
    buffer = _protocol_state["message_buffer"]
    lock = _protocol_state["message_lock"]
    with lock:
        messages = list(buffer)
        buffer.clear()
    if not messages:
        return {"tool": "check_messages", "status": "success", "attempt": "Checked messages", "stdout": "No new messages.", "stderr": ""}
    formatted = [f"[{msg.agent_id}] ({msg.topic}): {msg.payload}" for msg in messages]
    return {"tool": "check_messages", "status": "success", "attempt": f"Checked messages ({len(messages)} new)", "stdout": "\n".join(formatted), "stderr": ""}


def submit_task(arguments, work_dir=None):
    agent = _protocol_state["agent"]
    description = arguments["description"]
    payload = arguments.get("payload", {})
    task_payload = {"description": description, **payload}
    agent.submit_task(task_payload)
    return {
        "tool": "submit_task",
        "status": "success",
        "attempt": "Submitted task to work queue",
        "stdout": f"Task submitted: {description[:200]}",
        "stderr": "",
    }


def request_task(arguments, work_dir=None):
    import json
    import time as _time
    agent = _protocol_state["agent"]
    agent.request_task()
    # Give broker time to respond via DEALER
    _time.sleep(0.3)
    buffer = _protocol_state["message_buffer"]
    lock = _protocol_state["message_lock"]
    with lock:
        task_msgs = [m for m in buffer if m.message_type.value == "TASK_ASSIGN"]
        for m in task_msgs:
            buffer.remove(m)
    if task_msgs:
        assignment = task_msgs[0]
        if assignment.payload.get("status") == "no_tasks":
            return {"tool": "request_task", "status": "success", "attempt": "Requested task", "stdout": "No tasks available in the queue.", "stderr": ""}
        return {
            "tool": "request_task",
            "status": "success",
            "attempt": "Received task assignment",
            "stdout": f"Task assigned: {json.dumps(assignment.payload)}",
            "stderr": "",
        }
    return {"tool": "request_task", "status": "success", "attempt": "Requested task", "stdout": "No task assignment received yet.", "stderr": ""}


done_tool_definition = [
    {
        "name": "done",
        "description": "Call this when the task is complete. Provide a short summary of what was accomplished.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A short summary of what was accomplished.",
                },
            },
            "required": ["summary"],
        },
    },
]


def done(arguments, work_dir: str = None):
    return {
        "tool": "done",
        "status": "done",
        "attempt": "Task complete",
        "stdout": arguments["summary"],
        "stderr": "",
    }


plan_tool_definitions = [
    {
        "name": "plan",
        "description": "Enter planning mode. Explore the codebase with run_bash and read_file, then call submit_plan with your subtasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Brief description of what you're planning for.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "submit_plan",
        "description": "Submit your plan after exploring. Creates a tracked list of subtasks to execute.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Concrete, actionable subtask descriptions informed by exploration.",
                },
            },
            "required": ["subtasks"],
        },
    },
    {
        "name": "mark_complete",
        "description": "Mark the current subtask as complete and advance to the next one. Call after finishing each subtask.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished in this subtask.",
                },
            },
            "required": ["summary"],
        },
    },
]

_base_tools = (
    bash_tool_definitions
    + code_tool_definitions
    + sql_tool_definitions
    + web_tool_definitions
    + llm_tool_definitions
    + plan_tool_definitions
    + done_tool_definition
)
all_tools = _base_tools  # backward compat


def get_all_tools(protocol_enabled=False, work_queue_enabled=False):
    tools = list(_base_tools)
    if protocol_enabled:
        tools += messaging_tool_definitions
    if work_queue_enabled:
        tools += work_queue_tool_definitions
    return tools


TOOL_FUNCTIONS = {
    "run_bash": run_bash,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "sql_query": sql_query,
    "web_search": web_search,
    "fetch_url": fetch_url,
    "call_llm": call_llm,
    "done": done,
    "send_message": send_message,
    "check_messages": check_messages,
    "submit_task": submit_task,
    "request_task": request_task,
}


class Tool:
    def __init__(self, task, work_dir=None):
        self.task = task
        self.work_dir = work_dir

    def print_human_readable(self, data, action):
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"[{action}] {key}: {value}")
        elif isinstance(data, list):
            for item in data:
                print(f"[{action}] - {item}")
        else:
            print(f"[{action}] {data}")

    def run(self):
        function_name = self.task["function"]["name"]

        print(f"\n--- TOOL: {function_name.upper()} ---", flush=True)

        fn = TOOL_FUNCTIONS[function_name]
        self.print_human_readable(self.task["function"]["parameters"], function_name)
        return fn(self.task["function"]["parameters"], work_dir=self.work_dir)
