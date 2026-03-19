import json
import os
import re
from typing import Any, Dict

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


sql_tool_definitions = [
    {
        "name": "sql_query",
        "description": (
            "Execute a parameterized SQL query using SQLAlchemy. "
            "Read-only mode is enabled by default for safety."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query text. Use named bind params like :id.",
                },
                "params": {
                    "type": "object",
                    "description": "Named parameters for the SQL query.",
                },
                "db_url": {
                    "type": "string",
                    "description": (
                        "Optional SQLAlchemy database URL. "
                        "Defaults to SQL_DATABASE_URL or DATABASE_URL env vars."
                    ),
                },
                "read_only": {
                    "type": "boolean",
                    "description": (
                        "When true (default), only read-only SQL is allowed."
                    ),
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows to return for row-producing queries. Default: 200, max: 1000.",
                },
            },
            "required": ["query"],
        },
    },
]


_ALLOWED_READ_PREFIXES = {"select", "with", "show", "describe", "desc", "explain", "values"}
_WRITE_KEYWORD_RE = re.compile(
    r"\b(insert|update|delete|create|alter|drop|truncate|grant|revoke|merge|call|copy|vacuum|reindex|replace|attach|detach)\b",
    re.IGNORECASE,
)
_LEADING_COMMENT_RE = re.compile(r"^\s*(?:(?:--[^\n]*\n)|(?:/\*.*?\*/))*\s*", re.DOTALL)


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _first_keyword(query: str) -> str:
    cleaned = _LEADING_COMMENT_RE.sub("", query or "")
    match = re.match(r"([a-zA-Z]+)", cleaned)
    if not match:
        return ""
    return match.group(1).lower()


def _is_read_only_sql(query: str) -> bool:
    first = _first_keyword(query)
    if first not in _ALLOWED_READ_PREFIXES:
        return False
    if _WRITE_KEYWORD_RE.search(query or ""):
        return False
    return True


def _json_result(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def sql_query(arguments, work_dir=None):
    query = str(arguments.get("query", ""))
    if not query.strip():
        return {
            "tool": "sql_query",
            "status": "failure",
            "attempt": "Execute SQL query",
            "stdout": "",
            "stderr": "Argument 'query' is required.",
        }

    params = arguments.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return {
            "tool": "sql_query",
            "status": "failure",
            "attempt": "Execute SQL query",
            "stdout": "",
            "stderr": "Argument 'params' must be an object.",
        }

    read_only = bool(arguments.get("read_only", True))
    if read_only and not _is_read_only_sql(query):
        return {
            "tool": "sql_query",
            "status": "failure",
            "attempt": "Blocked non-read-only SQL",
            "stdout": "",
            "stderr": (
                "Blocked query in read-only mode. "
                "Allowed read prefixes: select, with, show, describe, desc, explain, values."
            ),
        }

    db_url = arguments.get("db_url") or os.environ.get("SQL_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        return {
            "tool": "sql_query",
            "status": "failure",
            "attempt": "Execute SQL query",
            "stdout": "",
            "stderr": "No database URL found. Provide 'db_url' or set SQL_DATABASE_URL/DATABASE_URL.",
        }

    max_rows = _clamp_int(arguments.get("max_rows", 200), 1, 1000, 200)
    op = _first_keyword(query).upper() or "SQL"

    engine = None
    try:
        engine = create_engine(db_url, future=True)
        with engine.connect() as conn:
            result = conn.execute(text(query), params)

            if result.returns_rows:
                rows = result.mappings().fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                if truncated:
                    rows = rows[:max_rows]
                payload = {
                    "rows": [dict(r) for r in rows],
                    "returned_rows": len(rows),
                    "truncated": truncated,
                    "max_rows": max_rows,
                }
                return {
                    "tool": "sql_query",
                    "status": "success",
                    "attempt": f"Executed {op} query in {'read-only' if read_only else 'read-write'} mode",
                    "stdout": _json_result(payload),
                    "stderr": "",
                }

            if read_only:
                return {
                    "tool": "sql_query",
                    "status": "failure",
                    "attempt": f"Executed {op} query in read-only mode",
                    "stdout": "",
                    "stderr": "Read-only mode requires a row-producing query.",
                }

            conn.commit()
            payload = {"rowcount": int(result.rowcount or 0)}
            return {
                "tool": "sql_query",
                "status": "success",
                "attempt": f"Executed {op} query in read-write mode",
                "stdout": _json_result(payload),
                "stderr": "",
            }
    except SQLAlchemyError as exc:
        return {
            "tool": "sql_query",
            "status": "failure",
            "attempt": f"Execute {op} query",
            "stdout": "",
            "stderr": f"SQL execution failed: {type(exc).__name__}: {exc}",
        }
    finally:
        if engine is not None:
            engine.dispose()
