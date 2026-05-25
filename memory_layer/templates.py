"""
Strategy template engine.

A *template* is a parametrized expression with named slots. Expanding a
template produces every concrete expression in the slot Cartesian product
(minus combinations excluded by constraints).

Slots support two value sources:
  - "literal" : explicit list of values
  - "catalogue" : pulled live from the WQ Brain catalogue snapshot
                  (filters by dataset, min coverage, min user_count, etc.)

Constraints are Python boolean expressions evaluated per-combo. Any combo
where a constraint evaluates True is **skipped** (think "skip_if").

Templates live as JSON files under private/templates/. Load all with
`load_all()` and expand a single template with `expand(template, ...)`.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE / "private" / "templates"
CATALOGUE_PATH = BASE / "memory_layer" / "brain_catalogue.json"


# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TemplateSlot:
    name: str
    type: str  # "literal" | "catalogue"
    values: List[Any] = field(default_factory=list)
    query: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyTemplate:
    id: str
    description: str
    form: str
    slots: List[TemplateSlot]
    concepts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    operators_hint: List[str] = field(default_factory=list)
    file_path: Optional[Path] = None


@dataclass
class ExpandedExpression:
    template_id: str
    expression: str
    bindings: Dict[str, Any]
    datafields: List[str]
    operators: List[str]
    concepts: List[str]
    settings: Dict[str, Any]


# ── loaders ──────────────────────────────────────────────────────────────────

def _load_catalogue() -> Dict[str, Any]:
    if not CATALOGUE_PATH.exists():
        return {}
    try:
        return json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_catalogue_slot(query: Dict[str, Any]) -> List[str]:
    """Resolve a `catalogue` slot into a list of datafield IDs."""
    cat = _load_catalogue()
    if not cat:
        return []
    setting_key = query.get("setting", "USA|1|TOP3000")
    rows = cat.get("datafields", {}).get(setting_key, [])
    dataset = query.get("dataset")
    min_cov = query.get("min_coverage", 0.8)
    min_users = query.get("min_users", 0)
    name_re = query.get("name_matches")
    name_pat = re.compile(name_re) if name_re else None
    limit = query.get("limit", 10)

    out = []
    for r in rows:
        if dataset and (r.get("dataset", {}) or {}).get("id") != dataset:
            continue
        if (r.get("coverage") or 0) < min_cov:
            continue
        if (r.get("userCount") or 0) < min_users:
            continue
        if name_pat and not name_pat.search(r.get("id", "")):
            continue
        out.append((r.get("userCount") or 0, r["id"]))

    out.sort(reverse=True)
    return [x[1] for x in out[:limit]]


def _parse_template_file(path: Path) -> StrategyTemplate:
    raw = json.loads(path.read_text(encoding="utf-8"))
    slots = []
    for name, s in raw.get("slots", {}).items():
        slots.append(TemplateSlot(
            name=name,
            type=s.get("type", "literal"),
            values=s.get("values", []),
            query=s.get("query", {}),
        ))
    return StrategyTemplate(
        id=raw["id"],
        description=raw.get("description", ""),
        form=raw["form"],
        slots=slots,
        concepts=raw.get("concepts", []),
        constraints=raw.get("constraints", []),
        settings=raw.get("settings", {}),
        operators_hint=raw.get("operators_hint", []),
        file_path=path,
    )


def load_all() -> List[StrategyTemplate]:
    if not TEMPLATES_DIR.exists():
        return []
    return [_parse_template_file(p) for p in sorted(TEMPLATES_DIR.glob("*.json"))]


def load(template_id: str) -> Optional[StrategyTemplate]:
    for t in load_all():
        if t.id == template_id:
            return t
    return None


# ── expansion ────────────────────────────────────────────────────────────────

def _slot_values(slot: TemplateSlot) -> List[Any]:
    if slot.type == "literal":
        return list(slot.values)
    if slot.type == "catalogue":
        return _resolve_catalogue_slot(slot.query)
    return []


def _extract_referenced_datafields(expression: str) -> List[str]:
    """Best-effort: find tokens that look like datafield references."""
    KNOWN_OPS = {
        "rank", "zscore", "scale", "winsorize", "quantile", "normalize",
        "group_rank", "group_zscore", "group_neutralize", "group_mean",
        "group_scale", "group_backfill",
        "ts_rank", "ts_mean", "ts_sum", "ts_delta", "ts_delay", "ts_decay_linear",
        "ts_av_diff", "ts_std_dev", "ts_corr", "ts_zscore", "ts_arg_max", "ts_arg_min",
        "ts_backfill", "kth_element", "last_diff_value", "days_from_last_change",
        "sign", "abs", "log", "exp", "sqrt", "power", "signed_power", "inverse",
        "add", "subtract", "multiply", "divide", "min", "max",
        "and", "or", "not", "equal", "not_equal", "greater", "less",
        "greater_equal", "less_equal", "if_else", "hump", "densify", "reverse",
        "industry", "subindustry", "sector", "market", "country",
    }
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expression)
    seen = []
    for t in tokens:
        if t.lower() in KNOWN_OPS:
            continue
        if t.isdigit():
            continue
        if t not in seen:
            seen.append(t)
    return seen


def _extract_operators(expression: str) -> List[str]:
    """Operators appear as identifiers immediately followed by '('."""
    return list(dict.fromkeys(re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expression)))


def _check_constraints(constraints: List[str], bindings: Dict[str, Any]) -> bool:
    """Evaluate all constraints. Return True if any constraint says SKIP."""
    for c in constraints:
        try:
            if bool(eval(c, {"__builtins__": {}}, dict(bindings))):
                return True
        except Exception:
            continue
    return False


def expand(template: StrategyTemplate, *,
           max_results: Optional[int] = None) -> List[ExpandedExpression]:
    """Return every concrete expression produced by the template."""
    slot_names = [s.name for s in template.slots]
    slot_values = [_slot_values(s) for s in template.slots]
    if any(not vs for vs in slot_values):
        return []

    out: List[ExpandedExpression] = []
    for combo in itertools.product(*slot_values):
        bindings = dict(zip(slot_names, combo))
        if _check_constraints(template.constraints, bindings):
            continue
        try:
            expr = template.form.format(**{k: str(v) for k, v in bindings.items()})
        except KeyError:
            continue
        expr = re.sub(r"\s+", " ", expr).strip()
        out.append(ExpandedExpression(
            template_id=template.id,
            expression=expr,
            bindings=bindings,
            datafields=_extract_referenced_datafields(expr),
            operators=_extract_operators(expr),
            concepts=template.concepts,
            settings=dict(template.settings),
        ))
        if max_results is not None and len(out) >= max_results:
            break
    return out


def stats(template: StrategyTemplate) -> Dict[str, Any]:
    """Compute the size + slot resolution without expanding fully."""
    sizes = []
    breakdown = []
    for s in template.slots:
        vs = _slot_values(s)
        sizes.append(max(1, len(vs)))
        breakdown.append({"slot": s.name, "type": s.type, "size": len(vs),
                          "sample": vs[:3]})
    total = 1
    for sz in sizes:
        total *= sz
    return {"total_combos": total, "slots": breakdown,
            "constraints": template.constraints}
