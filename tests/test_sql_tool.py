from __future__ import annotations

import json

from sqlalchemy import create_engine, text

from agent.tools.sql.sql_tool import sql_query


def _seed_db(db_url: str):
    engine = create_engine(db_url, future=True)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"))
            conn.execute(
                text("INSERT INTO users (id, name) VALUES (:id, :name)"),
                [{"id": 1, "name": "ada"}, {"id": 2, "name": "linus"}, {"id": 3, "name": "grace"}],
            )
    finally:
        engine.dispose()


def test_sql_query_parameterized_read_only_select(tmp_path):
    db_url = f"sqlite+pysqlite:///{tmp_path / 'test.db'}"
    _seed_db(db_url)

    out = sql_query(
        {
            "db_url": db_url,
            "query": "SELECT id, name FROM users WHERE id = :id",
            "params": {"id": 2},
        }
    )
    assert out["status"] == "success"
    payload = json.loads(out["stdout"])
    assert payload["returned_rows"] == 1
    assert payload["rows"][0]["name"] == "linus"


def test_sql_query_blocks_write_by_default(tmp_path):
    db_url = f"sqlite+pysqlite:///{tmp_path / 'test.db'}"
    _seed_db(db_url)

    out = sql_query(
        {
            "db_url": db_url,
            "query": "UPDATE users SET name = :name WHERE id = :id",
            "params": {"id": 1, "name": "ADA"},
        }
    )
    assert out["status"] == "failure"
    assert "read-only" in out["stderr"].lower()


def test_sql_query_allows_write_when_read_only_false(tmp_path):
    db_url = f"sqlite+pysqlite:///{tmp_path / 'test.db'}"
    _seed_db(db_url)

    out = sql_query(
        {
            "db_url": db_url,
            "read_only": False,
            "query": "UPDATE users SET name = :name WHERE id = :id",
            "params": {"id": 1, "name": "ADA"},
        }
    )
    assert out["status"] == "success"
    payload = json.loads(out["stdout"])
    assert payload["rowcount"] == 1

    verify = sql_query(
        {
            "db_url": db_url,
            "query": "SELECT name FROM users WHERE id = :id",
            "params": {"id": 1},
        }
    )
    verify_payload = json.loads(verify["stdout"])
    assert verify_payload["rows"][0]["name"] == "ADA"


def test_sql_query_enforces_params_object(tmp_path):
    db_url = f"sqlite+pysqlite:///{tmp_path / 'test.db'}"
    _seed_db(db_url)

    out = sql_query(
        {
            "db_url": db_url,
            "query": "SELECT id FROM users WHERE id = :id",
            "params": ["not", "a", "dict"],
        }
    )
    assert out["status"] == "failure"
    assert "params" in out["stderr"].lower()


def test_sql_query_truncates_rows(tmp_path):
    db_url = f"sqlite+pysqlite:///{tmp_path / 'test.db'}"
    _seed_db(db_url)

    out = sql_query(
        {
            "db_url": db_url,
            "query": "SELECT id, name FROM users ORDER BY id",
            "max_rows": 2,
        }
    )
    assert out["status"] == "success"
    payload = json.loads(out["stdout"])
    assert payload["returned_rows"] == 2
    assert payload["truncated"] is True
