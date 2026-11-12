"""Daily executive digest: sales + critical stock alerts, ready for Slack / WhatsApp.

Designed to run at 6:00 PM via cron::

    # crontab -e   (server time zone; or set CRON_TZ)
    CRON_TZ=America/Mexico_City
    0 18 * * * cd /opt/zero-hallucination-inventory-agent && /usr/bin/python3 daily_digest.py --send slack >> logs/digest.log 2>&1

Pipeline
1. **Extract** (deterministic, no LLM): the same read-only tools the agent
   uses pull today's sales per branch, critical stock alerts and dispatch
   exceptions.
2. **Write** (LLM): Claude turns the raw JSON into 4-6 executive bullets.
3. **Verify**: every number in the bullets is checked against the raw data
   (``grounding.py``). If the LLM output fails, or no API key is configured,
   a deterministic template summary is used instead, so the message that
   reaches management is never ungrounded.
4. **Format & deliver**: Slack Incoming Webhook (Block Kit) and WhatsApp
   Cloud API payloads; ``--send`` posts them, otherwise they are written to
   ``out/`` (dry run).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import config
from grounding import check_answer
from tools import execute_safe_read_only_sql, get_daily_sales_summary, get_low_stock_by_branch

DIGEST_SYSTEM = """You write the 6 PM operations digest for the leadership team of a 5-branch distributor.
You receive raw JSON extracted from the company database. Write 4 to 6 bullets, at most 110 words in total.

Rules:
- Use only values present in the JSON, copied exactly (same decimals). Do not compute percentages, averages,
  differences or forecasts; if a comparison is not in the data, leave it out.
- Cover, in this order: total revenue and the strongest/weakest branch; critical stock (how many SKUs, which
  branches, the most urgent SKU with its stock and minimum); delayed dispatches; anything with NO_RECORDS stated
  plainly as "no records".
- Each bullet starts with "• " and is one line. No title, no greeting, no closing."""


@dataclass
class Digest:
    day: str
    data: dict
    bullets: list[str]
    summary_source: str
    grounding: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return f"Daily Ops Digest | {self.day}"


# ----------------------------------------------------------------------------- 1. extract
def collect(day: str, db_path: Path = config.DB_PATH) -> dict:
    sales = get_daily_sales_summary(day, "ALL", db_path=db_path)
    critical = get_low_stock_by_branch("ALL", "critical", 50, db_path=db_path)
    dispatch = execute_safe_read_only_sql(
        "SELECT branch_id, status, COUNT(*) AS dispatches FROM dispatches "
        f"WHERE dispatch_date = '{sales.get('resolved_date', day)}' AND status IN ('DELAYED','PENDING','IN_TRANSIT') "
        "GROUP BY branch_id, status ORDER BY branch_id, status", db_path=db_path)
    delayed = sum(r["dispatches"] for r in dispatch["rows"] if r["status"] == "DELAYED")
    open_ = sum(r["dispatches"] for r in dispatch["rows"] if r["status"] != "DELAYED")
    return {
        "date": sales.get("resolved_date", day),
        "sales": {k: sales.get(k) for k in ("status", "totals", "rows", "top_skus", "available_date_range")
                  if sales.get(k) is not None},
        "critical_stock": {
            "status": critical["status"],
            "total_critical_skus": critical["total_matching"],
            "counts_by_branch": critical["counts_by_branch"],
            "branches_without_critical": [b for b in config.BRANCHES
                                          if b not in {c["branch_id"] for c in critical["counts_by_branch"]}],
            "most_urgent": critical["rows"][:5],
        },
        "dispatches": {"status": dispatch["status"], "delayed_total": delayed, "open_total": open_,
                       "by_branch_status": dispatch["rows"]},
    }


# ----------------------------------------------------------------------------- 2. write
def template_bullets(data: dict) -> list[str]:
    """Deterministic summary used in Demo mode and as the safety fallback."""
    b: list[str] = []
    s = data["sales"]
    if s["status"] == "NO_RECORDS":
        rng = s.get("available_date_range", {})
        b.append(f"• Sales: no records for {data['date']} (data available {rng.get('first')} to {rng.get('last')}).")
    else:
        rows = sorted(s["rows"], key=lambda r: r["revenue"], reverse=True)
        t = s["totals"]
        b.append(f"• Revenue ${t['revenue']:,.2f} across {len(rows)} branches ({t['units']:,} units, "
                 f"{t['transactions']:,} transactions).")
        b.append(f"• Strongest: Branch {rows[0]['branch_id']} ${rows[0]['revenue']:,.2f}. "
                 f"Weakest: Branch {rows[-1]['branch_id']} ${rows[-1]['revenue']:,.2f}.")
        top = s["top_skus"][0]
        b.append(f"• Top seller: {top['sku']} {top['name']} (${top['revenue']:,.2f}, {top['units']} units).")
    c = data["critical_stock"]
    if c["status"] == "NO_RECORDS":
        b.append("• Critical stock: no records, every branch is above 50% of its reorder point.")
    else:
        per = ", ".join(f"{x['branch_id']}: {x['low_stock_skus']}" for x in c["counts_by_branch"])
        ok = c["branches_without_critical"]
        b.append(f"• Critical stock: {c['total_critical_skus']} SKUs ({per})"
                 + (f"; no alerts in {', '.join(ok)}." if ok else "."))
        u = c["most_urgent"][0]
        b.append(f"• Most urgent: {u['sku']} {u['name']} at Branch {u['branch_id']}, stock {u['stock_on_hand']} "
                 f"vs min {u['min_stock_alert']}.")
    d = data["dispatches"]
    if d["status"] == "NO_RECORDS":
        b.append("• Dispatches: no records of delayed or open dispatches today.")
    else:
        b.append(f"• Dispatches: {d['delayed_total']} delayed, {d['open_total']} still pending or in transit.")
    return b


def write_summary(data: dict, backend: Any = None) -> tuple[list[str], str, dict]:
    if backend is None and not config.demo_mode():
        from agent import ClaudeBackend
        backend = ClaudeBackend()
    if backend is not None and backend.name != "mock":
        try:
            text = backend.complete(DIGEST_SYSTEM, json.dumps(data, default=str))
        except Exception as exc:  # network/API failure must not block the 6 PM report
            return template_bullets(data), f"template (LLM error: {type(exc).__name__})", {}
        bullets = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullets = [ln if ln.startswith("•") else "• " + ln.lstrip("-* ") for ln in bullets]
        report = check_answer("\n".join(bullets), [data])
        if bullets and report.grounded:
            return bullets, "llm", report.as_dict()
        return template_bullets(data), "template (LLM output failed grounding)", report.as_dict()
    bullets = template_bullets(data)
    return bullets, "template (demo mode)", check_answer("\n".join(bullets), [data]).as_dict()


# ----------------------------------------------------------------------------- 3. format
def to_text(d: Digest) -> str:
    return "\n".join([d.title, "", *d.bullets, "", f"Source: inventory DB, read-only | summary: {d.summary_source}"])


def to_slack(d: Digest) -> dict:
    """Slack Incoming Webhook payload (Block Kit + plain-text fallback)."""
    return {
        "text": f"{d.title}\n" + "\n".join(d.bullets),
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": d.title}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(d.bullets)}},
            {"type": "context", "elements": [{"type": "mrkdwn",
             "text": f"Read-only DB extract · numbers verified against source · summary: {d.summary_source}"}]},
        ],
    }


def to_whatsapp(d: Digest, to: str = config.WHATSAPP_TO) -> dict:
    """WhatsApp Cloud API text message payload (WhatsApp uses *bold*)."""
    body = f"*{d.title}*\n\n" + "\n".join(d.bullets) + "\n\n_Numbers verified against the inventory database._"
    return {"messaging_product": "whatsapp", "to": to or "<RECIPIENT_PHONE>", "type": "text",
            "text": {"preview_url": False, "body": body[:4096]}}


def post_json(url: str, payload: dict, token: str = "") -> int:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 (URL comes from operator config)
        return resp.status


# ----------------------------------------------------------------------------- orchestration
def build_digest(day: str = "today", db_path: Path = config.DB_PATH, backend: Any = None) -> Digest:
    day = date.today().isoformat() if day == "today" else day
    data = collect(day, db_path)
    bullets, source, grounding = write_summary(data, backend)
    return Digest(data["date"], data, bullets, source, grounding)


def main() -> None:
    ap = argparse.ArgumentParser(description="6 PM operations digest")
    ap.add_argument("--date", default="today", help="YYYY-MM-DD, 'today' (default) or 'latest'")
    ap.add_argument("--send", choices=["none", "slack", "whatsapp", "both"], default="none")
    ap.add_argument("--out", type=Path, default=config.OUT_DIR)
    args = ap.parse_args()

    digest = build_digest(args.date)
    payloads = {"slack": to_slack(digest), "whatsapp": to_whatsapp(digest)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"digest_{digest.day}.txt").write_text(to_text(digest), encoding="utf-8")
    (args.out / f"digest_{digest.day}.json").write_text(
        json.dumps({"digest": to_text(digest), "grounding": digest.grounding, "data": digest.data,
                    "payloads": payloads}, indent=2, default=str), encoding="utf-8")
    print(to_text(digest))

    targets = {"none": [], "slack": ["slack"], "whatsapp": ["whatsapp"], "both": ["slack", "whatsapp"]}[args.send]
    failed = False
    for t in targets:
        url, token = ((config.SLACK_WEBHOOK_URL, "") if t == "slack"
                      else (config.WHATSAPP_API_URL, config.WHATSAPP_TOKEN))
        if not url:
            print(f"[{t}] skipped: webhook URL not configured", file=sys.stderr)
            failed = True
            continue
        try:
            print(f"[{t}] delivered (HTTP {post_json(url, payloads[t], token)})", file=sys.stderr)
        except Exception as exc:
            print(f"[{t}] delivery failed: {exc}", file=sys.stderr)
            failed = True
    if not targets:
        print(f"\n[dry run] payloads written to {args.out}/digest_{digest.day}.json", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
