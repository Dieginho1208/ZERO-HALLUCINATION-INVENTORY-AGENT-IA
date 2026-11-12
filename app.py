"""Ops Copilot web console (Streamlit) for a fictional distributor, Kestrel Supply Co.

Shows how the agent looks to the people who use it: a branch/purchasing
manager asks questions in plain language, every answer carries a
"verified against database" badge and an audit trail of the tool calls,
and the same screen shows live stock alerts and the 6 PM digest preview.

    streamlit run app.py
    streamlit run app.py -- (then open ?demo=1 or ?demo=2 for a pre-filled conversation)
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
import seed_db
from agent import InventoryAgent
from daily_digest import build_digest, to_slack, to_whatsapp
from tools import get_daily_sales_summary, get_low_stock_by_branch

COMPANY = "Kestrel Supply Co."
USER = ("Laura Mendez", "Purchasing Manager")
DEMOS = {
    "1": ["Which SKUs are critically low in Branch D?", "Delayed dispatches today?"],
    "2": ["Any low stock in Branch C?", "What were sales on 2019-03-01?", "Delete all inventory records for Branch A"],
}
SUGGESTIONS = ["Critical stock in all branches", "Sales today in Branch A", "Delayed dispatches today?",
               "Any low stock in Branch C?", "Sales by category this month (sql)"]

st.set_page_config(page_title=f"{COMPANY} · Ops Copilot", page_icon="🦅", layout="wide")

st.markdown("""
<style>
:root { --brand:#0f3d5e; --accent:#f28c28; }
.block-container { padding-top: 1.2rem; }
header[data-testid="stHeader"] { display:none; }
.topbar { background: linear-gradient(90deg,#0f3d5e,#165a86); color:#fff; padding:14px 22px; border-radius:12px;
          display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
.topbar .brand { font-size:22px; font-weight:700; letter-spacing:.2px; }
.topbar .brand small { font-weight:400; opacity:.8; font-size:14px; margin-left:10px; }
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-left:6px; }
.pill.ro { background:#e6f4ea; color:#137333; } .pill.mode { background:#fff3e0; color:#a35200; }
.pill.ok { background:#e6f4ea; color:#137333; } .pill.warn { background:#fdecea; color:#b3261e; }
.pill.nr { background:#fff8e1; color:#8a6d00; }
.kpi { background:#fff; border:1px solid #e4e8ee; border-radius:12px; padding:14px 16px; }
.kpi .l { color:#5f6b7a; font-size:13px; } .kpi .v { font-size:26px; font-weight:700; color:#0f3d5e; }
.kpi .s { font-size:12px; color:#5f6b7a; }
.kpi.alert .v { color:#c62828; }
.digest { background:#fff; border:1px solid #e4e8ee; border-left:5px solid var(--accent); border-radius:10px;
          padding:16px 20px; font-size:15px; line-height:1.7; }
.phone { background:#eef2f6; border-radius:14px; padding:14px; border:1px solid #e4e8ee; }
.chathead { font-weight:700; color:#0f3d5e; margin-bottom:8px; } .chathead span { font-weight:400; color:#5f6b7a; font-size:12px; margin-left:6px; }
.chathead.small { font-weight:400; font-size:12px; color:#5f6b7a; margin:10px 0 0; }
.bubble { background:#fff; border-radius:4px 14px 14px 14px; padding:12px 14px; font-size:13.5px; line-height:1.6;
          box-shadow:0 1px 2px rgba(0,0,0,.08); }
.bubble .foot { margin-top:8px; font-size:12px; color:#137333; }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def ensure_db() -> str:
    if not config.DB_PATH.exists():
        seed_db.seed(config.DB_PATH)
    return str(config.DB_PATH)


ensure_db()
agent = InventoryAgent()
mode = "Demo mode" if agent.backend.name == "mock" else f"Claude · {config.ANTHROPIC_MODEL}"

st.markdown(f"""<div class="topbar"><div class="brand">🦅 {COMPANY}<small>Operations Copilot · 5 branches</small></div>
<div><span class="pill ro">🔒 Read-only data access</span><span class="pill ok">✔ Verified answers</span>
<span class="pill mode">{mode}</span></div></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- KPIs (straight from the tools)
sales = get_daily_sales_summary("latest", "ALL")
crit = get_low_stock_by_branch("ALL", "critical", 200)
low = get_low_stock_by_branch("ALL", "all", 200)
from tools import execute_safe_read_only_sql  # noqa: E402
delayed = execute_safe_read_only_sql(
    "SELECT COUNT(*) AS n FROM dispatches WHERE status='DELAYED' AND dispatch_date=(SELECT MAX(dispatch_date) FROM dispatches)")
k = st.columns(4)
cards = [
    ("Revenue today", f"${sales['totals']['revenue']:,.2f}", f"{sales['totals']['transactions']} transactions · {sales['resolved_date']}", ""),
    ("Critical SKUs", crit["total_matching"], "below 50% of reorder point", "alert"),
    ("Low-stock SKUs", low["total_matching"], "at or below reorder point", ""),
    ("Delayed dispatches", delayed["rows"][0]["n"], "today, all branches", "alert"),
]
for col, (label, val, sub, cls) in zip(k, cards):
    col.markdown(f'<div class="kpi {cls}"><div class="l">{label}</div><div class="v">{val}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### 👤 {USER[0]}\n{USER[1]} · Head office")
    st.divider()
    st.markdown("**Try asking**")
    for s in SUGGESTIONS:
        if st.button(s, use_container_width=True):
            st.session_state["pending"] = s.replace("Sales by category this month (sql)",
                "sql: SELECT p.category, ROUND(SUM(s.revenue),2) AS revenue FROM daily_sales s JOIN products p USING(sku) "
                "GROUP BY p.category ORDER BY revenue DESC")
    st.divider()
    st.caption("Every number in an answer is checked against the database before it is shown. "
               "If the data does not exist, the copilot says so instead of guessing.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["chat"] = []


def run(q: str) -> None:
    res = agent.ask(q)
    st.session_state["chat"].append({"q": q, "r": res.as_dict()})


if "chat" not in st.session_state:
    st.session_state["chat"] = []
    for q in DEMOS.get(st.query_params.get("demo", ""), []):
        run(q)

tab_chat, tab_stock, tab_digest = st.tabs(["💬 Ask the Copilot", "📦 Stock alerts", "📨 6 PM digest"])

# ---------------------------------------------------------------- chat
with tab_chat:
    for turn in st.session_state["chat"]:
        with st.chat_message("user", avatar="👩‍💼"):
            st.write(turn["q"])
        r = turn["r"]
        with st.chat_message("assistant", avatar="🦅"):
            g = r["grounding"]
            statuses = {t["status"] for t in r["tool_calls"]}
            if "REJECTED" in statuses:
                badge = '<span class="pill warn">⛔ Blocked by read-only policy · nothing changed</span>'
            elif statuses == {"NO_RECORDS"} or not r["tool_calls"]:
                badge = '<span class="pill nr">∅ No records found · nothing estimated</span>'
            elif g["grounded"]:
                badge = f'<span class="pill ok">✔ Verified · {g["numbers_checked"]} values checked against the database</span>'
            else:
                badge = '<span class="pill warn">⚠ Contains unverified values</span>'
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(r["answer"].replace(" | ", " · ").replace("CRITICAL", ":red[**CRITICAL**]")
                        .replace("DELAYED", ":orange[**DELAYED**]"))
            with st.expander(f"How I got this · {len(r['tool_calls'])} database call(s)"):
                for t in r["tool_calls"]:
                    st.markdown(f"**`{t['tool']}`** → `{t['status']}`" + (f" · {t['row_count']} rows" if t["row_count"] is not None else ""))
                    if "query" in t["arguments"]:
                        st.code(t["arguments"]["query"], language="sql")
                    else:
                        st.code(json.dumps(t["arguments"]), language="json")

    q = st.chat_input("Ask about stock, sales or dispatches…") or st.session_state.pop("pending", None)
    if q:
        run(q)
        st.rerun()

# ---------------------------------------------------------------- stock alerts
with tab_stock:
    c1, c2 = st.columns([1, 3])
    branch = c1.selectbox("Branch", ["ALL", *config.BRANCHES])
    sev = c1.radio("Severity", ["critical", "all"], horizontal=True)
    data = get_low_stock_by_branch(branch, sev, 200)
    by = pd.DataFrame(data["counts_by_branch"] or [], columns=["branch_id", "low_stock_skus"])
    with c1:
        st.markdown("**SKUs by branch**")
        st.bar_chart(by.set_index("branch_id") if not by.empty else by, color="#c62828", height=220)
    if data["status"] == "NO_RECORDS":
        c2.success(f"No records: no SKUs {sev} in {branch}.")
    else:
        df = pd.DataFrame(data["rows"])[["branch_id", "sku", "name", "category", "stock_on_hand", "min_stock_alert",
                                         "units_below_min", "severity"]]
        df.columns = ["Branch", "SKU", "Product", "Category", "On hand", "Reorder point", "Short by", "Severity"]
        c2.dataframe(df.style.map(lambda v: "color:#c62828;font-weight:600" if v == "CRITICAL" else "", subset=["Severity"]),
                     hide_index=True, use_container_width=True, height=430)

# ---------------------------------------------------------------- digest
with tab_digest:
    d = build_digest("latest")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"#### {d.title}")
        st.markdown('<div class="digest">' + "<br>".join(d.bullets) + "</div>", unsafe_allow_html=True)
        ok = d.grounding.get("grounded")
        st.markdown(f'<br><span class="pill {"ok" if ok else "warn"}">{"✔" if ok else "⚠"} '
                    f'{d.grounding.get("numbers_checked", 0)} values verified</span>'
                    f'<span class="pill mode">Summary: {d.summary_source}</span>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Delivered at 6:00 PM")
        preview = "<br>".join(d.bullets)
        st.markdown(f"""<div class="phone"><div class="chathead">🦅 Kestrel Ops Bot <span>#ops-leadership · today 6:00 PM</span></div>
<div class="bubble"><b>{d.title}</b><br><br>{preview}<div class="foot">✔ Numbers verified against the inventory database</div></div>
<div class="chathead small">Also sent by WhatsApp to 5 regional managers</div></div>""", unsafe_allow_html=True)
        with st.expander("Webhook payloads (chat + WhatsApp Cloud API)"):
            st.json({"chat_webhook": to_slack(d), "whatsapp": to_whatsapp(d)}, expanded=False)
