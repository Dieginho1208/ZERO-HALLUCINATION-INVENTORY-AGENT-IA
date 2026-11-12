"""Read-only SQL execution with defense in depth.

The LLM is allowed to write ad-hoc SQL, but it can never modify data, read
outside the business tables, or run a query that hangs the server. Four
independent layers have to be bypassed simultaneously for a write to land,
and each one is tested on its own in ``tests/test_sql_guard.py``:

1. **Static validation**: comments and string literals are stripped, then the
   statement must be a single ``SELECT``/``WITH`` with no write/DDL/PRAGMA/
   ATTACH keywords anywhere.
2. **Read-only connection**: the database file is opened with
   ``mode=ro`` (URI), so the SQLite engine itself refuses writes.
3. **``PRAGMA query_only``** on the same connection, as a second engine-level stop.
4. **SQLite authorizer callback**: every operation the compiled statement
   would perform is checked. Only ``SELECT``, column reads on allow-listed
   tables and safe functions are permitted; everything else is denied at
   prepare time.

Plus resource limits: a VM-step budget (progress handler) and a row cap.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import config

ALLOWED_TABLES = frozenset({"branches", "products", "inventory", "daily_sales", "dispatches"})

FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "REPLACE", "UPSERT", "MERGE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE", "GRANT",
    "REVOKE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "LOAD_EXTENSION",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
DENIED_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
                              "randomblob", "zeroblob"})
MAX_QUERY_CHARS = 5000

_SQLITE_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, _SQLITE_RECURSIVE}


class SQLGuardError(Exception):
    """Raised when a query is rejected. ``code`` is machine-readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _strip_literals_and_comments(sql: str) -> str:
    """Return a 'skeleton' of the SQL with comments removed and literals blanked.

    Keywords hidden inside strings (``WHERE note = 'DROP'``) must not trigger a
    false rejection, and keywords hidden inside comments must not slip through.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j
            out.append(" ")
        elif c == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j == -1:
                raise SQLGuardError("UNTERMINATED_COMMENT", "Unterminated /* comment.")
            i = j + 2
            out.append(" ")
        elif c in ("'", '"', "`", "["):
            close = "]" if c == "[" else c
            j = i + 1
            while True:
                j = sql.find(close, j)
                if j == -1:
                    raise SQLGuardError("UNTERMINATED_LITERAL", "Unterminated string or identifier.")
                if close != "]" and sql.startswith(close * 2, j):  # escaped quote
                    j += 2
                    continue
                break
            # String literals become an empty literal; quoted identifiers keep a placeholder name.
            out.append("''" if c == "'" else " _ident_ ")
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def validate_sql(sql: str) -> str:
    """Layer 1: static validation. Returns the cleaned query or raises ``SQLGuardError``."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLGuardError("EMPTY_QUERY", "Query is empty.")
    if len(sql) > MAX_QUERY_CHARS:
        raise SQLGuardError("QUERY_TOO_LONG", f"Query exceeds {MAX_QUERY_CHARS} characters.")
    skeleton = _strip_literals_and_comments(sql).strip()
    body = skeleton.rstrip(";").strip()
    if ";" in body:
        raise SQLGuardError("MULTIPLE_STATEMENTS", "Only a single statement is allowed.")
    first = body.split(None, 1)[0].upper() if body else ""
    if first not in ("SELECT", "WITH"):
        raise SQLGuardError("NOT_A_SELECT", "Only SELECT (or WITH ... SELECT) queries are allowed.")
    m = _FORBIDDEN_RE.search(body)
    if m:
        raise SQLGuardError("FORBIDDEN_KEYWORD", f"Keyword '{m.group(1).upper()}' is not allowed in read-only mode.")
    return sql.strip().rstrip(";").strip()


def _authorizer(action: int, arg1: Any, arg2: Any, _db: Any, _trigger: Any) -> int:
    """Layer 4: allow only reads on business tables."""
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ:
        table = (arg1 or "").lower()
        if table not in ALLOWED_TABLES:
            return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        fname = (arg2 or "").lower()
        if fname in DENIED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def open_read_only(db_path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    """Layers 2 + 3: an engine-level read-only connection."""
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}. Run `python seed_db.py` first.")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    conn.row_factory = sqlite3.Row
    return conn


def run_read_only_query(
    sql: str,
    params: tuple | dict = (),
    *,
    db_path: Path | str = config.DB_PATH,
    max_rows: int = config.SQL_MAX_ROWS,
    validate: bool = True,
) -> dict:
    """Execute a query through every guard layer and return a JSON-safe result.

    ``validate=False`` skips only layer 1; it exists so tests can prove the
    engine-level layers hold on their own. Application code never disables it.
    """
    query = validate_sql(sql) if validate else sql
    with closing(open_read_only(db_path)) as conn:
        steps = {"n": 0}

        def _budget() -> int:
            steps["n"] += 1
            return 1 if steps["n"] * 1000 > config.SQL_MAX_VM_STEPS else 0

        conn.set_progress_handler(_budget, 1000)
        conn.set_authorizer(_authorizer)
        try:
            cur = conn.execute(query, params)
            rows = cur.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as exc:
            msg = str(exc)
            if "interrupted" in msg:
                raise SQLGuardError("QUERY_TOO_EXPENSIVE", "Query aborted: exceeded execution budget.") from exc
            if "not authorized" in msg or "prohibited" in msg:
                raise SQLGuardError("NOT_AUTHORIZED", f"Blocked by read-only policy: {msg}") from exc
            if "readonly" in msg or "read-only" in msg:
                raise SQLGuardError("READ_ONLY", f"Blocked by read-only connection: {msg}") from exc
            raise SQLGuardError("SQL_ERROR", msg) from exc
        columns = [d[0] for d in cur.description or []]

    truncated = len(rows) > max_rows
    rows = rows[:max_rows]
    result = {
        "columns": columns,
        "rows": [dict(zip(columns, tuple(r))) for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }
    result["status"] = "OK" if rows else "NO_RECORDS"
    return result
