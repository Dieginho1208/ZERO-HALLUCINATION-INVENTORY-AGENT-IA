"""Post-generation grounding check: every number in an answer must be traceable.

Prompt rules reduce hallucinations; this module *verifies* them. After the LLM
writes its final answer we extract every numeric value (quantities, revenue,
SKU numbers, dates) and check it against the numbers that actually appeared in
the tool results of that conversation (plus the user's own question). Anything
that cannot be traced is reported as ungrounded, and the agent either asks the
model to rewrite or flags the answer, so a fabricated figure never reaches the
user silently.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

# 1,234,567.89 | 1234.5 | 42  (thousands separators optional)
_NUM_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
# Small integers are used in ordinary prose ("top 3", "2 branches"); always allowed.
ALWAYS_ALLOWED_MAX = 10


def _norm(intpart: str, frac: str | None) -> Decimal | None:
    try:
        return Decimal(intpart.replace(",", "") + (frac or ""))
    except InvalidOperation:  # pragma: no cover
        return None


def extract_numbers(text: str) -> list[Decimal]:
    out = []
    for m in _NUM_RE.finditer(text or ""):
        d = _norm(m.group(1), m.group(2))
        if d is not None:
            out.append(d)
    return out


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        yield len(obj)  # counting returned rows is a legitimate derived fact
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def evidence_numbers(tool_results: Iterable[dict], extra_text: str = "") -> set[Decimal]:
    nums: set[Decimal] = set()
    for res in tool_results:
        for leaf in _walk(res):
            if isinstance(leaf, bool) or leaf is None:
                continue
            if isinstance(leaf, (int, float)):
                nums.add(Decimal(str(leaf)))
                nums.update(extract_numbers(str(leaf)))
            elif isinstance(leaf, str):
                nums.update(extract_numbers(leaf))
    nums.update(extract_numbers(extra_text))
    return nums


def _matches(value: Decimal, evidence: set[Decimal]) -> bool:
    if value == value.to_integral_value() and 0 <= value <= ALWAYS_ALLOWED_MAX:
        return True
    if value in evidence:
        return True
    # Allow the model to round an evidenced value (12,345.67 -> 12,345.7 / 12,346 / 12.3k is NOT allowed).
    exp = value.as_tuple().exponent
    places = -exp if isinstance(exp, int) and exp < 0 else 0
    q = Decimal(1).scaleb(-places)
    return any(e.quantize(q) == value for e in evidence if abs(e - value) < 1)


@dataclass
class GroundingReport:
    grounded: bool
    ungrounded: list[str] = field(default_factory=list)
    checked: int = 0

    def as_dict(self) -> dict:
        return {"grounded": self.grounded, "ungrounded_values": self.ungrounded, "numbers_checked": self.checked}


def check_answer(answer: str, tool_results: list[dict], question: str = "") -> GroundingReport:
    evidence = evidence_numbers(tool_results, question)
    values = extract_numbers(answer)
    bad = []
    for v in values:
        if not _matches(v, evidence):
            s = format(v, "f")
            if s not in bad:
                bad.append(s)
    return GroundingReport(grounded=not bad, ungrounded=bad, checked=len(values))


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
