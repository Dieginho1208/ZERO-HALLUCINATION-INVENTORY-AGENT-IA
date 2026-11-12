"""Centralised configuration loaded from environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")


def _path(env_name: str, default: str) -> Path:
    p = Path(os.getenv(env_name, default))
    return p if p.is_absolute() else BASE_DIR / p


# --- LLM -------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
# Any current Claude model ID works here (e.g. claude-sonnet-5-5 for lower cost).
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5").strip()
# Reasoning depth. "low" keeps the agent terse and cheap; raise for harder analytics.
ANTHROPIC_EFFORT: str = os.getenv("ANTHROPIC_EFFORT", "medium").strip()
# Server-side refusal fallback ("default" = let the API pick a fallback model, "off" = disable).
ANTHROPIC_FALLBACKS: str = os.getenv("ANTHROPIC_FALLBACKS", "default").strip()
# NOTE on "temperature 0": current Claude models reject sampling parameters
# (temperature/top_p/top_k). Determinism in this project is enforced
# architecturally instead: tools are the only data source, tool schemas are
# strict, and every number in a final answer is verified against tool output
# by grounding.py before it reaches the user.

# --- Data ------------------------------------------------------------------
DB_PATH: Path = _path("INVENTORY_DB_PATH", "data/inventory.db")
OUT_DIR: Path = _path("DIGEST_OUT_DIR", "out")

# --- SQL guard-rails -------------------------------------------------------
SQL_MAX_ROWS: int = int(os.getenv("SQL_MAX_ROWS", "200"))
# SQLite VM instructions before a query is aborted (stops runaway cross joins).
SQL_MAX_VM_STEPS: int = int(os.getenv("SQL_MAX_VM_STEPS", "5000000"))

# --- Agent -----------------------------------------------------------------
MAX_TOOL_ROUNDS: int = int(os.getenv("MAX_TOOL_ROUNDS", "6"))

# --- Delivery ----------------------------------------------------------------
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "").strip()
WHATSAPP_API_URL: str = os.getenv("WHATSAPP_API_URL", "").strip()  # e.g. https://graph.facebook.com/v21.0/<PHONE_ID>/messages
WHATSAPP_TOKEN: str = os.getenv("WHATSAPP_TOKEN", "").strip()
WHATSAPP_TO: str = os.getenv("WHATSAPP_TO", "").strip()

BRANCHES = ("A", "B", "C", "D", "E")


def demo_mode() -> bool:
    """Demo/Mock mode is active whenever no API key is configured."""
    return not ANTHROPIC_API_KEY or os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")
