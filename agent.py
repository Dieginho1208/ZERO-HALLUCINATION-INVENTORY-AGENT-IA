"""Zero-hallucination operations agent (strict tool use over SQLite).

The model answers questions about inventory, sales and dispatches, but it has
no knowledge of the business except what the tools return in this
conversation. The control loop enforces that:

1. Claude receives the question plus three strict tools (see ``tools.py``).
2. Every tool call is validated, executed read-only, and returned as a JSON
   envelope with an explicit ``status`` (``OK`` / ``NO_RECORDS`` / ``REJECTED``).
3. The final answer goes through ``grounding.check_answer``: every number must
   appear in the tool results. If one does not, the model gets one chance to
   rewrite; if it still fails, the answer is flagged instead of trusted.

Run without ``ANTHROPIC_API_KEY`` to use the deterministic offline planner
(Demo/Mock mode), which exercises the exact same loop, tools and guards.

Usage::

    python agent.py "Which SKUs are critically low in Branch B?"
    python agent.py --trace "Delayed dispatches today?"
    python agent.py               # interactive
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import config
from grounding import GroundingReport, check_answer
from tools import TOOLS, anthropic_tools, dispatch_tool, to_json

SYSTEM_PROMPT = """You are the Operations Data Agent for a 5-branch industrial distributor (Branches A, B, C, D, E).
You answer questions about inventory, sales and dispatches for managers who make purchasing and logistics
decisions with your answers. A wrong number costs real money, so accuracy beats completeness.

<data_rules>
1. Your ONLY source of business facts is the output of the tools in this conversation. You have no other
   knowledge of this company's stock, sales, customers, drivers or prices.
2. Call a tool before stating any business fact. Prefer get_low_stock_by_branch and get_daily_sales_summary;
   use execute_safe_read_only_sql only when those cannot answer the question.
3. Every number, SKU, name, date and status you write must appear in a tool result. Copy values exactly as
   returned. Do not estimate, extrapolate, average or "round to make it look nice". If the user asks for a
   figure the tools did not return, compute it in SQL with execute_safe_read_only_sql instead of in your head.
4. If a tool returns status NO_RECORDS, say explicitly that there are no records for that request (name the
   branch/date/filter you checked) and stop. Do not fill the gap with typical or example values.
5. If a tool returns status REJECTED, explain briefly that the request could not be run and why. Never retry
   a rejected write or schema change in another form: you are strictly read-only.
6. If the question is outside inventory, sales and dispatches (competitor prices, forecasts, opinions about
   people), say that the database has no records to answer it.
7. If results are truncated or limited, say how many rows exist in total (total_matching) and how many you show.
</data_rules>

<style>
Answer in the user's language. Be brief: one sentence of direct answer, then bullets or a compact table when
listing items. Mention the branch and date the data refers to. No preamble, no closing offers.
</style>"""

CORRECTION_PROMPT = (
    "Verification failed: your answer contains values that do not appear in any tool result: {values}. "
    "Rewrite the answer using only values returned by the tools (call another tool if you need a figure). "
    "If the data is not available, say there are no records."
)


# ============================================================================ backends
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


class LLMBackend(Protocol):
    name: str

    def start(self, system: str, user_text: str) -> None: ...
    def step(self) -> Turn: ...
    def add_tool_results(self, results: list[tuple[ToolCall, dict]]) -> None: ...
    def add_user(self, text: str) -> None: ...
    def complete(self, system: str, prompt: str) -> str: ...


class ClaudeBackend:
    """Anthropic Messages API with strict client tools (manual agent loop)."""

    name = "claude"

    def __init__(self, client: Any = None, model: str = config.ANTHROPIC_MODEL,
                 effort: str = config.ANTHROPIC_EFFORT, fallbacks: str = config.ANTHROPIC_FALLBACKS):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.system = ""
        self.messages: list[dict] = []

    def _request(self, system: str, messages: list[dict], tools: list[dict] | None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": messages,
            "output_config": {"effort": self.effort},
            # Cache the stable prefix (tools + system prompt) across turns and runs.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "auto"}
        if self.fallbacks and self.fallbacks.lower() != "off":
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = self.fallbacks
        return self.client.beta.messages.create(**kwargs)

    def start(self, system: str, user_text: str) -> None:
        self.system = system
        self.messages = [{"role": "user", "content": user_text}]

    def step(self) -> Turn:
        resp = self._request(self.system, self.messages, anthropic_tools())
        # Append the full content (incl. thinking blocks) so the next request is a valid continuation.
        self.messages.append({"role": "assistant", "content": resp.content})
        turn = Turn(stop_reason=resp.stop_reason or "end_turn")
        if resp.stop_reason == "refusal":
            return turn
        for block in resp.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "tool_use":
                turn.tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
        return turn

    def add_tool_results(self, results: list[tuple[ToolCall, dict]]) -> None:
        # All results for one assistant turn go back in a single user message.
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": to_json(res),
             "is_error": res.get("status") == "REJECTED"}
            for call, res in results
        ]})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def complete(self, system: str, prompt: str) -> str:
        resp = self._request(system, [{"role": "user", "content": prompt}], None)
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class MockBackend:
    """Deterministic offline planner used in Demo/Mock mode and tests.

    It routes the question to tools with keyword rules and writes the answer
    only from tool results, so the full pipeline (tool dispatch, SQL guard,
    grounding check) runs exactly as in production, at zero cost.
    """

    name = "mock"

    _BRANCH_RE = re.compile(r"\b(?:branch|sucursal|store)\s+([A-Ea-e])\b", re.IGNORECASE)
    _DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

    def __init__(self) -> None:
        self.question = ""
        self.results: list[tuple[ToolCall, dict]] = []
        self.phase = "plan"

    def start(self, system: str, user_text: str) -> None:
        self.question = user_text.split("[Question]", 1)[-1].strip()
        self.results, self.phase = [], "plan"

    def add_tool_results(self, results: list[tuple[ToolCall, dict]]) -> None:
        self.results.extend(results)
        self.phase = "answer"

    def add_user(self, text: str) -> None:  # correction requests: the mock answer is already grounded
        self.phase = "answer"

    # -- planning ------------------------------------------------------------
    def _branch(self) -> str:
        m = self._BRANCH_RE.search(self.question)
        return m.group(1).upper() if m else "ALL"

    def _date(self) -> str:
        q = self.question.lower()
        m = self._DATE_RE.search(q)
        if m:
            return m.group(1)
        if "yesterday" in q or "ayer" in q:
            return (date.today() - timedelta(days=1)).isoformat()
        if "today" in q or "hoy" in q:
            return "today"
        return "latest"

    def _plan(self) -> list[ToolCall]:
        q = self.question.lower()
        branch, day = self._branch(), self._date()
        if q.startswith("sql:"):
            return [ToolCall("m1", "execute_safe_read_only_sql", {"query": self.question[4:].strip()})]
        if re.search(r"\b(delete|drop|update|reset|wipe|borra\w*|elimina\w*)\b", q):
            # Simulates a model that tries to act on a write request: the SQL guard must stop it.
            where = "" if branch == "ALL" else f" WHERE branch_id = '{branch}'"
            return [ToolCall("m1", "execute_safe_read_only_sql", {"query": f"DELETE FROM inventory{where}"})]
        if any(k in q for k in ("dispatch", "deliver", "driver", "route", "despacho", "entrega", "chofer", "ruta")):
            day_sql = ("(SELECT MAX(dispatch_date) FROM dispatches)" if day == "latest"
                       else f"'{date.today().isoformat() if day == 'today' else day}'")
            br = "" if branch == "ALL" else f" AND branch_id = '{branch}'"
            only_delayed = any(k in q for k in ("delay", "late", "retras", "atras"))
            status = " AND status = 'DELAYED'" if only_delayed else " AND status IN ('DELAYED','PENDING','IN_TRANSIT')"
            return [ToolCall("m1", "execute_safe_read_only_sql", {"query": (
                "SELECT dispatch_id, branch_id, dispatch_date, route, driver, customer, status FROM dispatches "
                f"WHERE dispatch_date = {day_sql}{br}{status} ORDER BY status, branch_id, dispatch_id")})]
        if any(k in q for k in ("stock", "inventor", "reorder", "agotad", "critical", "crític", "critic", "replenish")):
            critical = any(k in q for k in ("critical", "crític", "critic", "out of stock", "agotad"))
            return [ToolCall("m1", "get_low_stock_by_branch",
                             {"branch_id": branch, "severity": "critical" if critical else "all", "limit": 10})]
        if any(k in q for k in ("sale", "revenue", "sold", "venta", "vend", "factur", "ingreso")):
            return [ToolCall("m1", "get_daily_sales_summary", {"sale_date": day, "branch_id": branch})]
        return []

    # -- answering -----------------------------------------------------------
    def _answer(self) -> str:
        if not self.results:
            return ("I have no records that answer this question. I can only report inventory, sales and "
                    "dispatch data from the company database.")
        parts = []
        for call, res in self.results:
            if res.get("status") == "REJECTED":
                parts.append(f"The request could not be run: {res.get('error')} (read-only policy). No data was changed.")
                continue
            parts.append(getattr(self, f"_fmt_{call.name}")(res))
        return "\n\n".join(parts)

    @staticmethod
    def _where(branch: str) -> str:
        return "all branches" if branch == "ALL" else f"Branch {branch}"

    def _fmt_get_low_stock_by_branch(self, res: dict) -> str:
        a = res["arguments"]
        label = "critically low (below 50% of reorder point)" if a["severity"] == "critical" else "at or below the reorder point"
        if res["status"] == "NO_RECORDS":
            return f"No records: there are no SKUs {label} in {self._where(a['branch_id'])}."
        lines = [f"{res['total_matching']} SKUs are {label} in {self._where(a['branch_id'])}"
                 + (f" (showing {res['returned']})." if res["returned"] < res["total_matching"] else ".")]
        for r in res["rows"]:
            lines.append(f"- Branch {r['branch_id']} | {r['sku']} {r['name']} | stock {r['stock_on_hand']} "
                         f"/ min {r['min_stock_alert']} ({r['severity']})")
        if a["branch_id"] == "ALL" and res["counts_by_branch"]:
            lines.append("By branch: " + ", ".join(f"{c['branch_id']}: {c['low_stock_skus']}" for c in res["counts_by_branch"]))
        return "\n".join(lines)

    def _fmt_get_daily_sales_summary(self, res: dict) -> str:
        a = res["arguments"]
        if res["status"] == "NO_RECORDS":
            rng = res.get("available_date_range", {})
            return (f"No records: there are no sales for {self._where(a['branch_id'])} on {a['sale_date']}. "
                    f"Sales data is available from {rng.get('first')} to {rng.get('last')}.")
        t = res["totals"]
        lines = [f"Sales on {res['resolved_date']} for {self._where(a['branch_id'])}: ${t['revenue']:,.2f} revenue, "
                 f"{t['units']:,} units, {t['transactions']:,} transactions."]
        lines += [f"- Branch {r['branch_id']}: ${r['revenue']:,.2f} ({r['units']:,} units)" for r in res["rows"]]
        lines.append("Top SKUs: " + "; ".join(f"{s['sku']} {s['name']} ${s['revenue']:,.2f}" for s in res["top_skus"]))
        return "\n".join(lines)

    def _fmt_execute_safe_read_only_sql(self, res: dict) -> str:
        if res["status"] == "NO_RECORDS":
            return "No records: the query returned no rows."
        cols = res["columns"]
        if {"dispatch_id", "driver", "status"} <= set(cols):
            lines = [f"{res['row_count']} dispatches found:"]
            lines += [f"- {r['dispatch_id']} | Branch {r['branch_id']} | {r.get('route', '')} | driver {r['driver']} | "
                      f"{r.get('customer', '')} | {r['status']}" for r in res["rows"][:25]]
            return "\n".join(lines)
        head = f"{res['row_count']} rows" + (" (truncated)" if res.get("truncated") else "") + ":"
        body = ["- " + " | ".join(str(r[c]) for c in cols) for r in res["rows"][:25]]
        if res["row_count"] > 25:
            body.append(f"(showing 25 of {res['row_count']})")
        return "\n".join([head, *body])

    def step(self) -> Turn:
        if self.phase == "plan":
            self.phase = "answer"
            calls = self._plan()
            if calls:
                return Turn(tool_calls=calls, stop_reason="tool_use")
        return Turn(text=self._answer())

    def complete(self, system: str, prompt: str) -> str:
        return ""  # callers fall back to their deterministic template


def make_backend() -> LLMBackend:
    return MockBackend() if config.demo_mode() else ClaudeBackend()


# ============================================================================ agent
@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[dict]
    grounding: GroundingReport
    backend: str
    rounds: int

    def as_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "backend": self.backend, "rounds": self.rounds,
                "tool_calls": self.tool_calls, "grounding": self.grounding.as_dict()}


class InventoryAgent:
    def __init__(self, backend: LLMBackend | None = None, db_path: Path = config.DB_PATH,
                 max_rounds: int = config.MAX_TOOL_ROUNDS):
        self.backend = backend or make_backend()
        self.db_path = db_path
        self.max_rounds = max_rounds

    def ask(self, question: str) -> AgentResult:
        user_text = f"[Context] Today's date: {date.today().isoformat()}\n[Question] {question}"
        self.backend.start(SYSTEM_PROMPT, user_text)
        evidence: list[dict] = []
        trace: list[dict] = []
        corrections = 0
        answer = ""
        report = GroundingReport(grounded=True)

        for rounds in range(1, self.max_rounds + 1):
            turn = self.backend.step()
            if turn.stop_reason == "refusal":
                answer = "The request was declined by the model's safety system. No data was returned."
                break
            if turn.tool_calls:
                results = []
                for call in turn.tool_calls:
                    res = dispatch_tool(call.name, call.arguments, db_path=self.db_path)
                    results.append((call, res))
                    evidence.append(res)
                    trace.append({"tool": call.name, "arguments": call.arguments, "status": res.get("status"),
                                  "row_count": res.get("row_count")})
                self.backend.add_tool_results(results)
                continue
            answer = turn.text.strip()
            report = check_answer(answer, evidence, user_text)
            if not report.grounded and corrections < 1:
                corrections += 1
                self.backend.add_user(CORRECTION_PROMPT.format(values=", ".join(report.ungrounded)))
                continue
            break
        else:
            rounds = self.max_rounds
            answer = answer or "I could not complete this request within the tool-call budget. No answer was produced."

        if not report.grounded:
            answer += ("\n\n[Verification warning] These values could not be traced to database results and "
                       f"must not be relied on: {', '.join(report.ungrounded)}")
        return AgentResult(question, answer, trace, report, self.backend.name, rounds)


# ============================================================================ CLI
def _print(result: AgentResult, trace: bool) -> None:
    if trace:
        for t in result.tool_calls:
            print(f"  -> {t['tool']}({json.dumps(t['arguments'])}) => {t['status']} rows={t['row_count']}")
        g = result.grounding
        print(f"  -> grounding: {'PASS' if g.grounded else 'FAIL'} ({g.checked} numbers checked)\n")
    print(result.answer)


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-hallucination inventory agent")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--trace", action="store_true", help="show tool calls and grounding verdict")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()

    if not config.DB_PATH.exists():
        sys.exit("Database not found. Run `python seed_db.py` first.")
    agent = InventoryAgent()
    mode = "DEMO/MOCK (no API key)" if agent.backend.name == "mock" else f"Claude ({config.ANTHROPIC_MODEL})"
    print(f"[zero-hallucination agent | backend: {mode} | tools: {', '.join(t.name for t in TOOLS)}]\n", file=sys.stderr)

    questions = [" ".join(args.question)] if args.question else None
    while True:
        q = questions.pop() if questions else input("\nask> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        res = agent.ask(q)
        print(json.dumps(res.as_dict(), indent=2, default=str)) if args.json else _print(res, args.trace)
        if questions is not None and not questions:
            break


if __name__ == "__main__":
    main()
