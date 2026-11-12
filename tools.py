"""Business tools exposed to the LLM.

The model never sees the database directly. It can only call these functions,
and it can only state facts that come back from them. Each tool:

* has a strict JSON schema (``additionalProperties: false``, every field
  required, enums wherever possible) so arguments are validated before any
  query runs;
* returns a JSON envelope with an explicit ``status`` (``OK``,
  ``NO_RECORDS`` or ``REJECTED``) so "no data" is a first-class, unambiguous
  answer instead of an empty list the model might paper over.

Tool definitions are provider-neutral (``ToolSpec``) and render to both the
Anthropic Messages API and the OpenAI function-calling format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import config
from sql_guard import SQLGuardError, run_read_only_query

BRANCH_ENUM = ["ALL", *config.BRANCHES]

SCHEMA_DOC = """Tables (SQLite):
- branches(branch_id TEXT PK 'A'..'E', name, city)
- products(sku TEXT PK e.g. 'SKU-1001', name, category, unit, unit_price REAL, unit_cost REAL)
- inventory(branch_id, sku, stock_on_hand INT, min_stock_alert INT, last_restock_date TEXT ISO)
- daily_sales(sale_id, sale_date TEXT ISO 'YYYY-MM-DD', branch_id, sku, qty INT, unit_price REAL, revenue REAL)
- dispatches(dispatch_id TEXT PK, branch_id, dispatch_date TEXT ISO, customer, route, driver,
             status IN ('PENDING','IN_TRANSIT','DELIVERED','DELAYED','CANCELLED'), items_count INT, order_value REAL)
Categories: Fasteners, Electrical, Plumbing, Packaging, Safety, Tools, Lubricants, Cleaning."""


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool definition (JSON Schema parameters)."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]

    def to_anthropic(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.parameters, "strict": True}

    def to_openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters, "strict": True}}


def _envelope(tool: str, args: dict, rows: list[dict], **extra: Any) -> dict:
    out = {"tool": tool, "arguments": args, "status": "OK" if rows else "NO_RECORDS",
           "row_count": len(rows), "rows": rows}
    out.update(extra)
    if not rows:
        out["message"] = "No records matched this request. Do not estimate or invent values."
    return out


def _branch_filter(branch_id: str) -> tuple[str, tuple]:
    if branch_id == "ALL":
        return "", ()
    return " AND i.branch_id = ?", (branch_id,)


# --------------------------------------------------------------------------- tools
def get_low_stock_by_branch(branch_id: str, severity: str, limit: int, *, db_path: Path = config.DB_PATH) -> dict:
    """SKUs at or below their reorder point (``critical`` = below 50 % of it)."""
    args = {"branch_id": branch_id, "severity": severity, "limit": limit}
    if branch_id not in BRANCH_ENUM or severity not in ("all", "critical"):
        return {"tool": "get_low_stock_by_branch", "arguments": args, "status": "REJECTED",
                "error": "Invalid branch_id or severity."}
    limit = max(1, min(int(limit), 200))
    where, params = _branch_filter(branch_id)
    cond = "i.stock_on_hand * 2 < i.min_stock_alert" if severity == "critical" else "i.stock_on_hand <= i.min_stock_alert"
    base = f"FROM inventory i JOIN products p USING (sku) WHERE {cond}{where}"
    rows = run_read_only_query(
        f"""SELECT i.branch_id, i.sku, p.name, p.category, i.stock_on_hand, i.min_stock_alert,
                   (i.min_stock_alert - i.stock_on_hand) AS units_below_min,
                   CASE WHEN i.stock_on_hand * 2 < i.min_stock_alert THEN 'CRITICAL' ELSE 'LOW' END AS severity
            {base}
            ORDER BY (CAST(i.stock_on_hand AS REAL) / i.min_stock_alert), i.branch_id, i.sku
            LIMIT ?""",
        (*params, limit), db_path=db_path)["rows"]
    total = run_read_only_query(f"SELECT COUNT(*) AS n {base}", params, db_path=db_path)["rows"][0]["n"]
    by_branch = run_read_only_query(
        f"SELECT i.branch_id, COUNT(*) AS low_stock_skus {base} GROUP BY i.branch_id ORDER BY i.branch_id",
        params, db_path=db_path)["rows"]
    return _envelope("get_low_stock_by_branch", args, rows, total_matching=total,
                     returned=len(rows), counts_by_branch=by_branch,
                     severity_rule={"LOW": "stock_on_hand <= min_stock_alert",
                                    "CRITICAL": "stock_on_hand below 50% of min_stock_alert"})


def get_daily_sales_summary(sale_date: str, branch_id: str, *, db_path: Path = config.DB_PATH) -> dict:
    """Revenue, units and top SKUs for one day, per branch."""
    args = {"sale_date": sale_date, "branch_id": branch_id}
    if branch_id not in BRANCH_ENUM:
        return {"tool": "get_daily_sales_summary", "arguments": args, "status": "REJECTED", "error": "Invalid branch_id."}
    if sale_date == "latest":
        sale_date = run_read_only_query("SELECT MAX(sale_date) AS d FROM daily_sales", db_path=db_path)["rows"][0]["d"]
    elif sale_date == "today":
        sale_date = date.today().isoformat()
    try:
        date.fromisoformat(sale_date or "")
    except ValueError:
        return {"tool": "get_daily_sales_summary", "arguments": args, "status": "REJECTED",
                "error": "sale_date must be YYYY-MM-DD, 'today' or 'latest'."}

    where = "" if branch_id == "ALL" else " AND branch_id = ?"
    params = (sale_date,) if branch_id == "ALL" else (sale_date, branch_id)
    rows = run_read_only_query(
        f"""SELECT branch_id, ROUND(SUM(revenue), 2) AS revenue, SUM(qty) AS units,
                   COUNT(*) AS transactions
            FROM daily_sales WHERE sale_date = ?{where}
            GROUP BY branch_id ORDER BY branch_id""", params, db_path=db_path)["rows"]
    extra: dict[str, Any] = {"resolved_date": sale_date}
    if rows:
        extra["totals"] = {"revenue": round(sum(r["revenue"] for r in rows), 2),
                           "units": sum(r["units"] for r in rows),
                           "transactions": sum(r["transactions"] for r in rows)}
        extra["top_skus"] = run_read_only_query(
            f"""SELECT s.sku, p.name, SUM(s.qty) AS units, ROUND(SUM(s.revenue), 2) AS revenue
                FROM daily_sales s JOIN products p USING (sku)
                WHERE s.sale_date = ?{where.replace('branch_id', 's.branch_id')}
                GROUP BY s.sku ORDER BY revenue DESC LIMIT 3""", params, db_path=db_path)["rows"]
    else:
        rng = run_read_only_query("SELECT MIN(sale_date) AS first, MAX(sale_date) AS last FROM daily_sales",
                                  db_path=db_path)["rows"][0]
        extra["available_date_range"] = rng
    return _envelope("get_daily_sales_summary", {**args, "sale_date": sale_date}, rows, **extra)


def execute_safe_read_only_sql(query: str, *, db_path: Path = config.DB_PATH) -> dict:
    """Run an ad-hoc SELECT through the read-only guard."""
    args = {"query": query}
    try:
        res = run_read_only_query(query, db_path=db_path)
    except SQLGuardError as exc:
        return {"tool": "execute_safe_read_only_sql", "arguments": args, "status": "REJECTED",
                "error_code": exc.code, "error": exc.message}
    return _envelope("execute_safe_read_only_sql", args, res["rows"], columns=res["columns"],
                     truncated=res["truncated"])


# --------------------------------------------------------------------------- registry
TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="get_low_stock_by_branch",
        description=("List SKUs whose stock_on_hand is at or below min_stock_alert (reorder point). "
                     "severity='critical' returns only SKUs below 50% of the reorder point. "
                     "Use branch_id='ALL' for every branch. Returns rows, total_matching and counts_by_branch."),
        parameters={
            "type": "object",
            "properties": {
                "branch_id": {"type": "string", "enum": BRANCH_ENUM, "description": "Branch letter or ALL."},
                "severity": {"type": "string", "enum": ["all", "critical"]},
                "limit": {"type": "integer", "description": "Max rows to return (1-200). Use 20 if unsure."},
            },
            "required": ["branch_id", "severity", "limit"],
            "additionalProperties": False,
        },
        fn=get_low_stock_by_branch,
    ),
    ToolSpec(
        name="get_daily_sales_summary",
        description=("Sales for a single day: revenue, units and transactions per branch, totals and top 3 SKUs. "
                     "sale_date is 'YYYY-MM-DD', 'today' or 'latest' (most recent day with data). "
                     "If the day has no sales, status is NO_RECORDS and available_date_range is returned."),
        parameters={
            "type": "object",
            "properties": {
                "sale_date": {"type": "string", "description": "YYYY-MM-DD, 'today' or 'latest'."},
                "branch_id": {"type": "string", "enum": BRANCH_ENUM},
            },
            "required": ["sale_date", "branch_id"],
            "additionalProperties": False,
        },
        fn=get_daily_sales_summary,
    ),
    ToolSpec(
        name="execute_safe_read_only_sql",
        description=("Run ONE read-only SQLite SELECT for questions the other tools cannot answer "
                     "(dispatches, drivers, routes, trends, categories). Writes, PRAGMA, ATTACH and multiple "
                     "statements are rejected. Results are capped at "
                     f"{config.SQL_MAX_ROWS} rows; aggregate in SQL instead of fetching raw rows.\n" + SCHEMA_DOC),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SELECT or WITH ... SELECT statement."}},
            "required": ["query"],
            "additionalProperties": False,
        },
        fn=execute_safe_read_only_sql,
    ),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def anthropic_tools() -> list[dict]:
    return [t.to_anthropic() for t in TOOLS]


def openai_tools() -> list[dict]:
    return [t.to_openai() for t in TOOLS]


def dispatch_tool(name: str, arguments: dict, *, db_path: Path = config.DB_PATH) -> dict:
    """Validate arguments against the schema's keys and run the tool. Never raises."""
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return {"tool": name, "status": "REJECTED", "error": f"Unknown tool '{name}'."}
    props = spec.parameters["properties"]
    if not isinstance(arguments, dict) or set(arguments) != set(spec.parameters["required"]):
        return {"tool": name, "arguments": arguments, "status": "REJECTED",
                "error": f"Arguments must be exactly {sorted(props)}."}
    try:
        return spec.fn(**arguments, db_path=db_path)
    except SQLGuardError as exc:
        return {"tool": name, "arguments": arguments, "status": "REJECTED", "error_code": exc.code, "error": exc.message}
    except Exception as exc:  # tool failures are reported to the model, never hidden
        return {"tool": name, "arguments": arguments, "status": "REJECTED", "error": f"{type(exc).__name__}: {exc}"}


def to_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
