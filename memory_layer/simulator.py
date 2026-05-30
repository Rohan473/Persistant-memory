"""
WQ Brain simulation runner.

Submits an alpha expression, polls until it completes, parses the metrics,
and optionally writes results back to a markdown alpha file under
private/nodes/alphas/.

Respects a daily submission budget tracked in memory_layer/sim_usage.json.

Dependencies are intentionally minimal: only requests (via brain_api.py)
and frontmatter (only at write-back time).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

USAGE_PATH = Path(__file__).parent / "sim_usage.json"
ALPHAS_DIR = Path(__file__).resolve().parent.parent / "private" / "nodes" / "alphas"


# ── pipeline state machine ──────────────────────────────────────────────────
# Stages (mirrors xiegengcai's INIT/SIMULATED/SYNC/CHECKED/SUBMITTED but adds
# REJECTED + ACTIVE_OS for our reality):
#
#   INIT         template expanded but not yet submitted to WQB
#   SIMULATED    sim complete, metrics back, but gates not yet IS-passed
#   GATE_FAIL    sim complete; at least one IS check FAIL (most common end state)
#   IS_PASS      all IS gates pass; eligible to submit (user must click)
#   SUBMITTED    user has submitted; status went UNSUBMITTED -> ACTIVE; OS pending
#   ACTIVE_OS    OS checks resolving (weeks to months — see reference-wqb-os-check-resolution)
#   REJECTED     terminal failure (sim errored, expression invalid, etc.)
#
# This field is for AI/dashboard convenience — the source of truth for "what next"
# is still (status × grade × failure_modes × OS checks). See `derive_pipeline_state`.


PIPELINE_INIT       = "INIT"
PIPELINE_SIMULATED  = "SIMULATED"
PIPELINE_GATE_FAIL  = "GATE_FAIL"
PIPELINE_IS_PASS    = "IS_PASS"
PIPELINE_SUBMITTED  = "SUBMITTED"
PIPELINE_ACTIVE_OS  = "ACTIVE_OS"
PIPELINE_REJECTED   = "REJECTED"


def _derive_pipeline_state_after_sim(result, failure_modes) -> str:
    """Map a freshly-completed simulation to a pipeline_state value."""
    if not getattr(result, "succeeded", False):
        return PIPELINE_REJECTED
    return PIPELINE_GATE_FAIL if failure_modes else PIPELINE_IS_PASS


def derive_pipeline_state(metadata: dict) -> str:
    """Derive pipeline_state from existing alpha frontmatter — for backfill and
    for resync against WQB API responses. Idempotent on already-set states.

    Local `status` semantics: 'submitted' = passed our IS gate, 'rejected' = failed it.
    True submission (user clicked Submit at WQB) is recorded via `api_status` (filled
    by `scripts/sync_pipeline_states.py` or similar resync against /alphas/{id})."""
    api_status = (metadata.get("api_status") or metadata.get("wqb_status") or "").upper()
    if api_status == "ACTIVE":
        return PIPELINE_ACTIVE_OS
    if api_status in ("UNSUBMITTED", ""):  # not yet verified or known-unsubmitted
        pass
    status = (metadata.get("status") or "").lower()
    failure_modes = metadata.get("failure_modes") or []
    sharpe = metadata.get("sharpe")
    if sharpe is None:
        return PIPELINE_INIT
    if status == "submitted" and not failure_modes:
        # Passed local IS gate. Until api_status is verified ACTIVE, treat as IS_PASS,
        # not SUBMITTED — the user may not have clicked Submit at WQB yet.
        return PIPELINE_IS_PASS
    return PIPELINE_GATE_FAIL if failure_modes else PIPELINE_IS_PASS


# ── budget ───────────────────────────────────────────────────────────────────

class BudgetExhausted(Exception):
    pass


class QuietHours(Exception):
    """Raised when current time is inside the configured quiet window."""


@dataclass
class DailyBudget:
    """Per-UTC-day counter + optional local-time quiet-hours window."""
    limit: int = 30
    path: Path = USAGE_PATH
    quiet_hours_before_midnight: int = 3  # local time; 0 disables
    quiet_use_utc: bool = False           # if True, quiet window is UTC-based

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"date": self._today(), "count": 0}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("date") != self._today():
            return {"date": self._today(), "count": 0}
        return data

    def quiet_window(self):
        """Return (is_quiet: bool, message: str) for the current moment."""
        if self.quiet_hours_before_midnight <= 0:
            return False, ""
        from datetime import datetime as _dt, timedelta as _td
        if self.quiet_use_utc:
            now = _dt.now(timezone.utc).replace(tzinfo=None)
            tz_label = "UTC"
        else:
            now = _dt.now()
            tz_label = "local"
        next_midnight = (now.replace(hour=0, minute=0, second=0, microsecond=0)
                         + _td(days=1))
        cutoff = next_midnight - _td(hours=self.quiet_hours_before_midnight)
        if now >= cutoff:
            return True, (
                f"quiet hours active: now {now:%H:%M} {tz_label} "
                f">= cutoff {cutoff:%H:%M} {tz_label} "
                f"({self.quiet_hours_before_midnight}h before midnight)"
            )
        return False, ""

    def remaining(self) -> int:
        if self.limit <= 0:
            return 10**9  # unlimited sentinel
        return max(0, self.limit - self._load()["count"])

    def check(self) -> int:
        """Raise QuietHours / BudgetExhausted if blocked; return current count.
        Set limit <= 0 to disable the daily-budget check entirely (WQB-side cap
        and concurrency limit still apply)."""
        is_quiet, msg = self.quiet_window()
        if is_quiet:
            raise QuietHours(
                f"{msg}. Pass --ignore-quiet-hours to override."
            )
        if self.limit <= 0:
            return self._load()["count"]
        data = self._load()
        if data["count"] >= self.limit:
            raise BudgetExhausted(
                f"Daily submission budget exhausted ({data['count']}/{self.limit}). "
                f"Resets at next UTC midnight."
            )
        return data["count"]

    def increment(self) -> int:
        """Increment counter and persist. Call only AFTER a successful submission."""
        data = self._load()
        data["count"] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data["count"]

    def check_and_increment(self) -> int:
        """Deprecated: increments before submission, over-charges on failure."""
        self.check()
        return self.increment()

    def refund(self) -> int:
        """Decrement by 1 (never below 0). For correcting failed-submission charges."""
        data = self._load()
        data["count"] = max(0, data["count"] - 1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data["count"]


# ── simulation result ────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    sim_id: str
    expression: str
    status: str
    settings: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    yearly: List[Dict[str, Any]] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    alpha_id_remote: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True only if the sim finished AND all checks passed (= submittable alpha)."""
        if self.status.upper() not in ("COMPLETE", "COMPLETED", "SUCCESS", "WARNING"):
            return False
        return not self.failure_modes

    @property
    def failure_modes(self) -> List[str]:
        """Map failed WQ Brain check names → this project's failure-mode tags."""
        # WQ Brain check names observed: LOW_SHARPE, LOW_FITNESS, LOW_TURNOVER,
        # HIGH_TURNOVER, CONCENTRATED_WEIGHT, LOW_SUB_UNIVERSE_SHARPE,
        # SELF_CORRELATION, MATCHES_COMPETITION.
        NAME_MAP = {
            "LOW_SHARPE":              "low_sharpe",
            "LOW_FITNESS":             "low_fitness",
            "LOW_TURNOVER":            "low_turnover",
            "HIGH_TURNOVER":           "high_turnover",
            "LOW_SUB_UNIVERSE_SHARPE": "sub_universe_failure",
            "SELF_CORRELATION":        "correlated",
            "CONCENTRATED_WEIGHT":     "concentrated_weight",
            "MATCHES_COMPETITION":     "competition_match_failure",
        }
        modes = set()
        for chk in self.checks:
            res = (chk.get("result") or "").upper()
            if res not in ("FAIL", "ERROR"):
                continue
            raw = (chk.get("name") or "").upper()
            tag = NAME_MAP.get(raw)
            if tag:
                modes.add(tag)
        return sorted(modes)


# ── core run logic ───────────────────────────────────────────────────────────

def _coalesce(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _extract_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pull standardized metrics from an Alpha resource.

    WQ Brain stores raw decimals (turnover=0.0255, returns=0.0043). The rest
    of this codebase stores percentages (50.02%) and ‱ (basis points × 100),
    so this function performs the unit conversions on the way out.
    """
    src = payload.get("is") or payload.get("inSample") or payload.get("metrics") or payload
    if not isinstance(src, dict):
        return {}
    sharpe   = _coalesce(src, "sharpe", "Sharpe")
    fitness  = _coalesce(src, "fitness", "Fitness")
    turnover = _coalesce(src, "turnover", "Turnover")
    returns  = _coalesce(src, "returns", "Returns", "annualReturn")
    drawdown = _coalesce(src, "drawdown", "maxDrawdown", "DD")
    margin   = _coalesce(src, "margin", "Margin", "marginInBps")

    def pct(v):
        return None if v is None else round(v * 100.0, 4)
    def myriad(v):
        return None if v is None else round(v * 10000.0, 4)

    return {
        "sharpe":   sharpe,
        "fitness":  fitness,
        "turnover": pct(turnover),
        "returns":  pct(returns),
        "drawdown": pct(drawdown),
        "margin":   myriad(margin),
        "long_count":  _coalesce(src, "longCount", "long_count"),
        "short_count": _coalesce(src, "shortCount", "short_count"),
        "pnl":         _coalesce(src, "pnl"),
        "book_size":   _coalesce(src, "bookSize"),
    }


def _extract_yearly(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Try common shapes for yearly breakdown. Returns [] if not present."""
    src = (payload.get("is") or payload.get("inSample") or {})
    candidates = (
        src.get("yearly") or src.get("yearlyStats") or
        src.get("byYear") or payload.get("yearly") or []
    )
    out = []
    if isinstance(candidates, list):
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            yr = _coalesce(entry, "year", "yr")
            try:
                yr = int(yr)
            except (TypeError, ValueError):
                continue
            out.append({
                "year":     yr,
                "sharpe":   _coalesce(entry, "sharpe"),
                "fitness":  _coalesce(entry, "fitness"),
                "turnover": _coalesce(entry, "turnover"),
                "returns":  _coalesce(entry, "returns"),
                "drawdown": _coalesce(entry, "drawdown"),
                "margin":   _coalesce(entry, "margin"),
            })
    return out


def _parse_response(
    sim_id: str,
    expression: str,
    sim_payload: Dict[str, Any],
    alpha_payload: Optional[Dict[str, Any]] = None,
) -> SimulationResult:
    """
    WQ Brain returns the simulation envelope (status + resulting alpha id)
    separately from the alpha resource which holds the metrics. When
    alpha_payload is supplied (recommended), pull metrics + checks from it.
    """
    status = str(sim_payload.get("status") or sim_payload.get("state") or "UNKNOWN")
    src = alpha_payload or sim_payload
    settings = src.get("settings") or sim_payload.get("settings") or {}
    is_block = src.get("is") if isinstance(src.get("is"), dict) else {}
    checks = is_block.get("checks") or src.get("checks") or []
    return SimulationResult(
        sim_id=sim_id,
        expression=expression,
        status=status,
        settings=settings,
        metrics=_extract_metrics(src),
        yearly=_extract_yearly(src),
        checks=checks if isinstance(checks, list) else [],
        alpha_id_remote=sim_payload.get("alpha") or src.get("id"),
        raw={"simulation": sim_payload, "alpha": alpha_payload},
    )


def run_simulation(
    client,
    expression: str,
    settings: Optional[Dict[str, Any]] = None,
    *,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
    budget: Optional[DailyBudget] = None,
    on_progress=None,
) -> SimulationResult:
    """Submit → poll → parse. Raises BudgetExhausted if daily cap is reached."""
    if budget is None:
        budget = DailyBudget()
    budget.check()  # raise if exhausted, but don't increment yet

    if on_progress:
        on_progress(f"[{budget.remaining()}/{budget.limit} remaining] submitting expression ({len(expression)} chars)")

    sim_id = client.submit_simulation(expression, settings=settings)
    count_after = budget.increment()  # only count successful submissions
    if on_progress:
        on_progress(f"  sim_id = {sim_id}")

    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        payload = client.get_simulation(sim_id)
        status = str(payload.get("status") or payload.get("state") or "UNKNOWN").upper()
        if status != last_status:
            if on_progress:
                on_progress(f"  status: {status}")
            last_status = status
        if status in ("COMPLETE", "COMPLETED", "SUCCESS", "WARNING",
                      "ERROR", "FAIL", "FAILED"):
            alpha_payload = None
            alpha_id = payload.get("alpha")
            if alpha_id:
                try:
                    r = client._http.get(
                        f"{client.base_url}/alphas/{alpha_id}", timeout=client.timeout
                    )
                    if r.status_code < 400:
                        alpha_payload = r.json()
                        if on_progress:
                            on_progress(f"  alpha resource: /alphas/{alpha_id}")
                except Exception as e:
                    if on_progress:
                        on_progress(f"  WARN: could not fetch alpha {alpha_id}: {e}")
            return _parse_response(sim_id, expression, payload, alpha_payload)
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Simulation {sim_id} did not finish within {timeout}s "
        f"(last status: {last_status})"
    )


# ── write-back to markdown ───────────────────────────────────────────────────

def _normalize_expression(expr: str) -> str:
    """Collapse whitespace so equivalent formulas compare equal."""
    expr = re.sub(r"#[^\n]*", "", expr)         # strip line comments
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def _next_alpha_id(alphas_dir: Path = ALPHAS_DIR) -> str:
    nums = []
    for f in alphas_dir.glob("alpha_*.md"):
        m = re.match(r"alpha_(\d+)\.md$", f.name)
        if m:
            nums.append(int(m.group(1)))
    return f"alpha_{(max(nums) + 1) if nums else 0:04d}"


def find_matching_alpha(
    expression: str,
    alphas_dir: Path = ALPHAS_DIR,
    settings: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Find an alpha .md whose normalized expression matches AND whose key
    settings (universe, delay, neutralization, decay, truncation) match.
    Same expression with different settings = different experiment.
    """
    import frontmatter
    target = _normalize_expression(expression)
    # Normalize settings keys (API uses upper-case neutralization, markdown lower)
    def _key(meta_or_settings):
        s = meta_or_settings or {}
        return (
            str(s.get("universe", "")).upper(),
            int(s.get("delay", 1)) if s.get("delay") is not None else 1,
            str(s.get("neutralization", "")).upper(),
            (int(s.get("decay")) if s.get("decay") is not None else None),
            (float(s.get("truncation")) if s.get("truncation") is not None else None),
        )
    target_key = _key(settings) if settings else None
    for f in sorted(alphas_dir.glob("alpha_*.md")):
        try:
            post = frontmatter.load(str(f))
            existing = _normalize_expression(post.metadata.get("expression") or "")
            if not existing or existing != target:
                continue
            if target_key is None:
                return f  # expression-only match (legacy behavior)
            if _key(post.metadata) == target_key:
                return f
        except Exception:
            continue
    return None


def _yearly_table_md(yearly: List[Dict[str, Any]]) -> str:
    if not yearly:
        return ""
    lines = [
        "| Year | Sharpe | Turnover | Fitness | Returns | Drawdown | Margin |",
        "|------|--------|----------|---------|---------|----------|--------|",
    ]
    def fmt(v, spec):
        if v is None:
            return "n/a"
        try:
            return format(v, spec)
        except (TypeError, ValueError):
            return "n/a"
    for y in sorted(yearly, key=lambda r: r["year"]):
        lines.append(
            f"| {y['year']} | "
            f"{fmt(y.get('sharpe'),  '+.2f')} | "
            f"{fmt(y.get('turnover'), '.2f')}% | "
            f"{fmt(y.get('fitness'), '+.2f')} | "
            f"{fmt(y.get('returns'), '+.2f')}% | "
            f"{fmt(y.get('drawdown'), '.2f')}% | "
            f"{fmt(y.get('margin'),  '+.2f')}‱ |"
        )
    return "\n".join(lines)


def write_back(
    result: SimulationResult,
    *,
    hypothesis: str = "",
    concepts: Optional[List[str]] = None,
    datafields: Optional[List[str]] = None,
    operators: Optional[List[str]] = None,
    parent_alpha: Optional[str] = None,
    alphas_dir: Path = ALPHAS_DIR,
    force_new: bool = False,
) -> Path:
    """Write or update a markdown alpha file from a SimulationResult."""
    import frontmatter

    match_path = None if force_new else find_matching_alpha(
        result.expression, alphas_dir, settings=result.settings
    )

    # SAFETY: if this sim ERRORED (no metrics back), refuse to overwrite an
    # existing alpha file. Otherwise we wipe verified metrics with NULLs
    # whenever a settings-only variant errors. Create a fresh ERROR-stub file
    # instead so the failure is recorded without clobbering history.
    if match_path and not result.succeeded and not (result.metrics or {}).get("sharpe"):
        match_path = None
        force_new = True

    metrics = result.metrics
    settings = result.settings or {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    failure_modes = result.failure_modes  # already empty if all checks passed

    def _rating(s):
        if s is None: return "Needs Improvement"
        if s >= 1.58: return "Good"
        if s >= 1.25: return "Average"
        return "Needs Improvement"

    if match_path:
        post = frontmatter.load(str(match_path))
        alpha_id = post.metadata.get("id", match_path.stem)
    else:
        alpha_id = _next_alpha_id(alphas_dir)
        post = frontmatter.Post("")
        post.metadata = {
            "id": alpha_id,
            "expression": result.expression,
            "datafields": datafields or [],
            "operators": operators or [],
            "concepts": concepts or [],
            "parent_alpha": parent_alpha or "null",
            "correlated_with": [],
            "hypothesis": hypothesis,
            "created": today,
        }

    # Common updates (overwrite metric fields whether new or existing)
    sharpe = metrics.get("sharpe")
    post.metadata.update({
        "expression": result.expression,
        "universe": settings.get("universe", post.metadata.get("universe", "TOP3000")),
        "region":   settings.get("region",   post.metadata.get("region", "USA")),
        "delay":    settings.get("delay",    post.metadata.get("delay", 1)),
        "neutralization": settings.get("neutralization",
                                       post.metadata.get("neutralization", "SUBINDUSTRY")).lower(),
        "decay":      settings.get("decay",      post.metadata.get("decay")),
        "truncation": settings.get("truncation", post.metadata.get("truncation")),
        "sharpe":   sharpe,
        "fitness":  metrics.get("fitness"),
        "turnover": metrics.get("turnover"),
        "returns":  metrics.get("returns"),
        "drawdown": metrics.get("drawdown"),
        "margin":   metrics.get("margin"),
        "status":   "submitted" if result.succeeded else "rejected",
        "rating":   _rating(sharpe),
        "failure_modes": failure_modes,
        "simulation_id": result.sim_id,
        "remote_alpha_id": result.alpha_id_remote,
        "last_simulated": today,
        # Pipeline state — explicit machine-readable lifecycle stage. Derived from
        # (succeeded, failure_modes); upgraded later by the submitter / OS poll.
        # See `derive_pipeline_state` for the rules.
        "pipeline_state": _derive_pipeline_state_after_sim(result, failure_modes),
    })

    # Body
    body_lines = [
        f"{'Updated' if match_path else 'Created'} from WQ Brain simulation `{result.sim_id}` on {today}.",
        f"Status: **{result.status}**.",
    ]
    if failure_modes:
        body_lines.append(f"Failure modes inferred: {', '.join(failure_modes)}.")
    if result.yearly:
        body_lines.append("")
        body_lines.append("**Year-by-year breakdown:**")
        body_lines.append(_yearly_table_md(result.yearly))
    if metrics.get("sharpe") is not None:
        body_lines.append("")
        body_lines.append(
            f"**Aggregate:** Sharpe {metrics.get('sharpe')} · "
            f"Turnover {metrics.get('turnover')}% · "
            f"Fitness {metrics.get('fitness')} · "
            f"Returns {metrics.get('returns')}% · "
            f"Drawdown {metrics.get('drawdown')}% · "
            f"Margin {metrics.get('margin')}‱"
        )
    post.content = "\n".join(body_lines)

    out_path = match_path or (alphas_dir / f"{alpha_id}.md")
    alphas_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(frontmatter.dumps(post), encoding="utf-8")

    # Auto-rebuild graph so dashboards / queries see the new alpha immediately.
    # Runs in a detached background process; doesn't block write_back's caller.
    try:
        import subprocess, sys as _sys
        build_script = alphas_dir.resolve().parent.parent.parent / "scripts" / "build_graph.py"
        if build_script.exists():
            subprocess.Popen(
                [_sys.executable, str(build_script)],
                cwd=str(build_script.parent.parent),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0,
            )
    except Exception:
        pass  # silently skip — write_back's primary job (markdown write) already succeeded

    return out_path
