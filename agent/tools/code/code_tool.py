import os


code_tool_definitions = [
    {
        "name": "read_file",
        "description": "Read the contents of a file and return them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if they don't exist. Overwrites existing files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Make a surgical edit to a file by replacing an exact string match. The old_string must appear exactly once in the file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace. Must match exactly once in the file. Include enough surrounding context to be unambiguous.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace old_string with. Can be empty to delete the matched text.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
]


def read_file(arguments, work_dir: str = None):
    path = arguments["path"]
    if work_dir and not os.path.isabs(path):
        path = os.path.join(work_dir, path)
    with open(path, "r") as f:
        content = f.read()
    return {
        "tool": "read_file",
        "status": "success",
        "attempt": f"Read file {path}",
        "stdout": content,
        "stderr": "",
    }


def write_file(arguments, work_dir: str = None):
    path = arguments["path"]
    content = arguments["content"]
    if work_dir and not os.path.isabs(path):
        path = os.path.join(work_dir, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return {
        "tool": "write_file",
        "status": "success",
        "attempt": f"Wrote file {path}",
        "stdout": f"Wrote {len(content)} bytes to {path}",
        "stderr": "",
    }


def edit_file(arguments, work_dir: str = None):
    path = arguments["path"]
    old_string = arguments["old_string"]
    new_string = arguments["new_string"]
    if work_dir and not os.path.isabs(path):
        path = os.path.join(work_dir, path)

    with open(path, "r") as f:
        content = f.read()

    count = content.count(old_string)
    if count == 0:
        return {
            "tool": "edit_file",
            "status": "failure",
            "attempt": f"Edit {path}",
            "stdout": "",
            "stderr": f"old_string not found in {path}",
        }
    if count > 1:
        return {
            "tool": "edit_file",
            "status": "failure",
            "attempt": f"Edit {path}",
            "stdout": "",
            "stderr": f"old_string appears {count} times in {path} — must be unique. Provide more surrounding context.",
        }

    new_content = content.replace(old_string, new_string, 1)
    with open(path, "w") as f:
        f.write(new_content)

    return {
        "tool": "edit_file",
        "status": "success",
        "attempt": f"Edited {path}",
        "stdout": f"Replaced {len(old_string)} chars with {len(new_string)} chars in {path}",
        "stderr": "",
    }
