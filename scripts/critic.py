"""
Fresh-context critic agent. Invoked from Claude Code via `!python scripts/critic.py`.

Every invocation starts with a clean slate:
  - No chat-transcript context (does not read ~/.claude/projects/.../*.jsonl)
  - No auto-memory context (does not read ~/.claude/projects/.../memory/*.md)
  - Only the actual project state: recent templates, alpha results, git log, external research

Runs two layers:
  1. Local heuristic checks (always — no API key needed)
  2. Multi-model LLM critique — Claude, GPT, and Gemini in parallel, each given
     the same fresh state and the same critic prompt. Different models surface
     different blind spots; cross-model agreement = stronger signal.

Usage:
  !python scripts/critic.py                       # heuristics + all available models
  !python scripts/critic.py --models claude,gpt   # subset
  !python scripts/critic.py --no-llm              # heuristics only
  !python scripts/critic.py --since 2026-05-27    # restrict to artifacts modified since date

Env vars consumed (each LLM is skipped if its key is missing):
  ANTHROPIC_API_KEY  -> Claude
  OPENAI_API_KEY     -> GPT
  GEMINI_API_KEY or GOOGLE_API_KEY -> Gemini
  OPENCODE_API_KEY   -> OpenCode Zen (OpenAI-compatible gateway, https://opencode.ai/zen/v1)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Force UTF-8 stdout so LLM responses with greek letters / math symbols / emoji
# don't crash on Windows (default cp1252 can't encode them).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent

CRITIC_SYSTEM = """You are a skeptical, independent reviewer of recent work on a quantitative-alpha research project (WorldQuant Brain platform).

Your role: question assumptions, verify correctness, call out misguides. You have NO prior context — you must form opinions from the artifacts provided alone.

Your checks should cover:
  1. UNVERIFIED ASSUMPTIONS — claims or labels that were inferred but not validated (e.g., "X is the best model" without comparison against baseline; field-name guesses; model interpretations)
  2. MISSING CONTROLS — out-of-sample tests, sign-flip checks, correlation against existing portfolio, robustness to settings
  3. OVERCONFIDENT FRAMINGS — claims of "near submission gate" / "breakthrough" / "near optimal" that aren't supported by the data
  4. INCONSISTENCIES — numbers / rankings claimed in template descriptions vs what alpha files show
  5. OVERFITTING RISK — too-specific signal optimizations that may not generalize
  6. SKIPPED VERIFICATION — places where a sanity check should have been run but wasn't
  7. LOGICAL ERRORS — flawed inferences, misapplied statistics, wrong direction of effect

Format: terse bullet points. Cite filenames and specific numbers. No filler praise. If you don't have enough info to critique something, say so."""

# --------- context gathering ---------

def _read_text(p: Path, max_chars: int = 5000) -> str:
    try:
        s = p.read_text(encoding="utf-8", errors="replace")
        return s[:max_chars] + (f"\n...[truncated, {len(s)} chars total]" if len(s) > max_chars else "")
    except Exception as e:
        return f"[read error: {e}]"


def _since_cutoff(since: str | None) -> float:
    if not since:
        return (datetime.now() - timedelta(days=2)).timestamp()
    try:
        return datetime.fromisoformat(since).timestamp()
    except Exception:
        return (datetime.now() - timedelta(days=2)).timestamp()


def collect_state(since: str | None) -> dict:
    """Read project artifacts. Explicitly skips chat transcript and auto-memory."""
    cutoff = _since_cutoff(since)
    state = {}

    # Recent templates
    tmpl_dir = BASE / "private" / "templates"
    templates = []
    if tmpl_dir.exists():
        for f in sorted(tmpl_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.stat().st_mtime >= cutoff:
                templates.append({"name": f.name, "content": _read_text(f, 3000)})
    state["templates"] = templates

    # Recent alpha results
    alphas_dir = BASE / "private" / "nodes" / "alphas"
    alphas = []
    if alphas_dir.exists():
        for f in sorted(alphas_dir.glob("alpha_*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.stat().st_mtime >= cutoff:
                alphas.append({"name": f.name, "content": _read_text(f, 1500)})
        alphas = alphas[:60]  # cap
    state["alphas"] = alphas

    # External research note list
    research_dir = BASE / "external_research"
    research = []
    if research_dir.exists():
        for f in research_dir.glob("*"):
            if f.is_file():
                research.append({"name": f.name, "size_bytes": f.stat().st_size})
    state["external_research"] = research

    # Git log of recent commits
    try:
        gitlog = subprocess.check_output(
            ["git", "log", "--oneline", "-n", "20"], cwd=str(BASE), text=True, errors="replace"
        ).strip()
    except Exception as e:
        gitlog = f"[git log failed: {e}]"
    state["git_log"] = gitlog

    # Git status (uncommitted changes)
    try:
        gitstatus = subprocess.check_output(
            ["git", "status", "--short"], cwd=str(BASE), text=True, errors="replace"
        ).strip()
    except Exception as e:
        gitstatus = f"[git status failed: {e}]"
    state["git_status"] = gitstatus

    return state


# --------- heuristic checks ---------

def heuristic_checks(state: dict) -> list[str]:
    findings = []

    # H1. Scientific-notation literal in templates (WQB rejects 1e-N).
    for t in state["templates"]:
        # match standalone 1e-N (not preceded by decimal). 0.000001 is fine.
        if re.search(r"(?<![0-9.])\d+e-?\d+(?![0-9.])", t["content"]):
            findings.append(
                f"H1. Template `{t['name']}` contains scientific-notation literal "
                f"(WQB returns status=ERROR on these — verified empirically in this project)"
            )

    # H2. Submission-gate sanity: do any recent alphas pass Sharpe>=1.25 AND Fitness>=1.15?
    passing = []
    metrics_by_alpha = []
    for a in state["alphas"]:
        sh = re.search(r"^sharpe:\s*([-\d.]+)\s*$", a["content"], re.M)
        fi = re.search(r"^fitness:\s*([-\d.]+)\s*$", a["content"], re.M)
        tu = re.search(r"^turnover:\s*([-\d.]+)\s*$", a["content"], re.M)
        if sh and fi and tu:
            s = float(sh.group(1)); f = float(fi.group(1)); t = float(tu.group(1))
            metrics_by_alpha.append((a["name"], s, f, t))
            if s >= 1.25 and f >= 1.15:
                passing.append(a["name"])
    if metrics_by_alpha and not passing:
        best_sharpe = max(metrics_by_alpha, key=lambda r: r[1])
        best_fitness = max(metrics_by_alpha, key=lambda r: r[2])
        findings.append(
            f"H2. Submission gate (Sharpe>=1.25, Fitness>=1.15) NOT reached in {len(metrics_by_alpha)} "
            f"recent alphas. Best Sharpe={best_sharpe[1]:.2f} ({best_sharpe[0]}); "
            f"best Fitness={best_fitness[2]:.2f} ({best_fitness[0]}). Treat 'close to gate' claims with skepticism."
        )

    # H3. Concentrated-weight flag prevalence
    cw_count = sum(1 for a in state["alphas"] if "concentrated_weight" in a["content"])
    if cw_count >= 3:
        findings.append(
            f"H3. `concentrated_weight` failure mode appears on {cw_count} recent alphas. "
            f"May be structural to the signal class — needs position-sizing fix, not more signal tuning."
        )

    # H4. Sub-universe failure prevalence
    suf_count = sum(1 for a in state["alphas"] if "sub_universe_failure" in a["content"])
    if suf_count >= 5:
        findings.append(
            f"H4. `sub_universe_failure` flag on {suf_count} recent alphas. Coverage gaps in "
            f"some industries — may inflate apparent Sharpe by trading only well-covered names."
        )

    # H5. Templates referencing new datasets without external_research/ entries
    referenced_datasets = set()
    for t in state["templates"]:
        c = t["content"].lower()
        for ds in ("model53", "mdl53", "annualized_pd", "mdl77", "fnd2", "fundamental6", "scl12"):
            if ds in c:
                referenced_datasets.add(ds)
    research_names = " ".join(r["name"].lower() for r in state["external_research"])
    missing = []
    for ds in referenced_datasets:
        if ds not in research_names and ds.replace("mdl53", "model") not in research_names:
            # Loose match — also accept "pd" as proxy for model53
            if ds in ("model53", "mdl53", "annualized_pd") and "pd" not in research_names and "credit" not in research_names:
                missing.append(ds)
            elif ds not in ("model53", "mdl53", "annualized_pd") and ds not in research_names:
                missing.append(ds)
    if missing:
        findings.append(
            f"H5. Templates reference dataset(s) {sorted(set(missing))} but no matching "
            f"file in external_research/. Literature-first rule may have been skipped."
        )

    # H6. Duplicate expressions across templates (waste / preflight false-positives)
    seen = {}
    for t in state["templates"]:
        # Extract form / expr values from JSON
        try:
            data = json.loads(t["content"].split("...[truncated")[0])
            form = data.get("form", "")
            if "{" not in form and form:
                seen.setdefault(form, []).append(t["name"])
            slots = data.get("slots", {})
            expr_slot = slots.get("expr", {}) if isinstance(slots, dict) else {}
            for v in (expr_slot.get("values", []) if isinstance(expr_slot, dict) else []):
                seen.setdefault(v, []).append(t["name"])
        except Exception:
            continue
    dupes = {k: v for k, v in seen.items() if len(set(v)) > 1}
    if dupes:
        n = len(dupes)
        sample = next(iter(dupes.items()))
        findings.append(
            f"H6. {n} expression(s) duplicated across multiple templates "
            f"(e.g., `{sample[0][:80]}...` appears in {sample[1]}). "
            f"Settings-only variations may trigger preflight false-positives — pass --force or filter."
        )

    return findings


# --------- LLM critics (multi-model) ---------

def _format_state_for_llm(state: dict, max_alphas: int = 30) -> str:
    parts = ["# PROJECT STATE FOR FRESH-CONTEXT CRITIC\n"]
    parts.append("## Recent templates (assistant-authored alpha proposals)\n")
    for t in state["templates"][:15]:
        parts.append(f"### {t['name']}\n```json\n{t['content']}\n```\n")
    parts.append("\n## Recent alpha simulation results (status / metrics / expression)\n")
    for a in state["alphas"][:max_alphas]:
        parts.append(f"### {a['name']}\n```yaml\n{a['content']}\n```\n")
    parts.append(f"\n## External research notes available\n")
    if state["external_research"]:
        for r in state["external_research"]:
            parts.append(f"- {r['name']} ({r['size_bytes']} bytes)\n")
    else:
        parts.append("(none)\n")
    parts.append(f"\n## Recent git commits\n```\n{state['git_log']}\n```\n")
    parts.append(f"\n## Git status (uncommitted changes)\n```\n{state['git_status']}\n```\n")
    parts.append("\n---\n\nNow apply your fresh-context critic role on the above. Be specific. Bullet points only.")
    return "".join(parts)


def critique_claude(state: dict, model: str = "claude-haiku-4-5-20251001") -> tuple[str, str]:
    """Return (label, output)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return ("Claude", "[skipped — ANTHROPIC_API_KEY not set]")
    try:
        from anthropic import Anthropic
    except ImportError:
        return ("Claude", "[skipped — `pip install anthropic` required]")
    try:
        client = Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=CRITIC_SYSTEM,
            messages=[{"role": "user", "content": _format_state_for_llm(state)}],
        )
        text = resp.content[0].text if resp.content else "(empty)"
        return (f"Claude ({model})", text)
    except Exception as e:
        return ("Claude", f"[error: {type(e).__name__}: {e}]")


def critique_gpt(state: dict, model: str = "gpt-4o-mini") -> tuple[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return ("GPT", "[skipped — OPENAI_API_KEY not set]")
    try:
        from openai import OpenAI
    except ImportError:
        return ("GPT", "[skipped — `pip install openai` required]")
    try:
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=2000,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": _format_state_for_llm(state)},
            ],
        )
        text = resp.choices[0].message.content or "(empty)"
        return (f"GPT ({model})", text)
    except Exception as e:
        return ("GPT", f"[error: {type(e).__name__}: {e}]")


def critique_opencode(state: dict, model: str = "claude-sonnet-4-5") -> tuple[str, str]:
    """OpenAI-compatible gateway (https://opencode.ai/zen/v1)."""
    key = os.environ.get("OPENCODE_API_KEY")
    if not key:
        return ("OpenCode", "[skipped — OPENCODE_API_KEY not set]")
    try:
        from openai import OpenAI
    except ImportError:
        return ("OpenCode", "[skipped — `pip install openai` required]")
    try:
        client = OpenAI(api_key=key, base_url="https://opencode.ai/zen/v1")
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=2000,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": _format_state_for_llm(state)},
            ],
        )
        text = resp.choices[0].message.content or "(empty)"
        return (f"OpenCode ({model})", text)
    except Exception as e:
        return ("OpenCode", f"[error: {type(e).__name__}: {e}]")


def critique_deepseek(state: dict, model: str = "deepseek-chat") -> tuple[str, str]:
    """OpenAI-compatible — https://api.deepseek.com/v1. Models: deepseek-chat, deepseek-reasoner."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return ("DeepSeek", "[skipped — DEEPSEEK_API_KEY not set]")
    try:
        from openai import OpenAI
    except ImportError:
        return ("DeepSeek", "[skipped — `pip install openai` required]")
    try:
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=2000,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": _format_state_for_llm(state)},
            ],
        )
        text = resp.choices[0].message.content or "(empty)"
        return (f"DeepSeek ({model})", text)
    except Exception as e:
        return ("DeepSeek", f"[error: {type(e).__name__}: {e}]")


def critique_gemini(state: dict, model: str = "gemini-2.0-flash-exp") -> tuple[str, str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return ("Gemini", "[skipped — GEMINI_API_KEY/GOOGLE_API_KEY not set]")
    try:
        from google import genai
    except ImportError:
        return ("Gemini", "[skipped — `pip install google-genai` required]")
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=model,
            contents=CRITIC_SYSTEM + "\n\n---\n\n" + _format_state_for_llm(state),
        )
        text = getattr(resp, "text", None) or "(empty)"
        return (f"Gemini ({model})", text)
    except Exception as e:
        return ("Gemini", f"[error: {type(e).__name__}: {e}]")


MODEL_RUNNERS = {
    "claude": critique_claude,
    "gpt": critique_gpt,
    "gemini": critique_gemini,
    "opencode": critique_opencode,
    "deepseek": critique_deepseek,
}


# --------- main ---------

def main():
    ap = argparse.ArgumentParser(description="Fresh-context multi-model critic agent")
    ap.add_argument("--models", default="claude,gpt,gemini,opencode,deepseek",
                    help="Comma-separated subset of {claude,gpt,gemini,opencode,deepseek}. Default: all.")
    ap.add_argument("--deepseek-model", default=None,
                    help="Override DeepSeek model (default: deepseek-chat; alt: deepseek-reasoner)")
    ap.add_argument("--opencode-model", default=None,
                    help="Override the model name sent to OpenCode Zen (default: claude-sonnet-4-5)")
    ap.add_argument("--no-llm", action="store_true", help="Heuristic checks only — skip all LLM calls")
    ap.add_argument("--since", default=None,
                    help="ISO date — only artifacts modified on/after (default: last 48h)")
    args = ap.parse_args()

    print("=" * 72)
    print("CRITIC AGENT — fresh context, no chat memory, no auto-memory")
    print("=" * 72)

    state = collect_state(args.since)
    print(f"\nLoaded state: {len(state['templates'])} template(s), "
          f"{len(state['alphas'])} recent alpha(s), "
          f"{len(state['external_research'])} research file(s)")
    print(f"Since: {args.since or '(last 48h)'}")

    # Heuristic layer
    print("\n" + "-" * 72)
    print("HEURISTIC CHECKS (local, no LLM)")
    print("-" * 72)
    findings = heuristic_checks(state)
    if findings:
        for f in findings:
            print(f"  • {f}\n")
    else:
        print("  (no heuristic violations detected)")

    if args.no_llm:
        print("\n[--no-llm set, skipping LLM critique]")
        return

    # LLM layer — multi-model in parallel
    requested = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    runners = [(name, MODEL_RUNNERS[name]) for name in requested if name in MODEL_RUNNERS]
    if not runners:
        print(f"\nNo valid models in --models='{args.models}'. Available: {list(MODEL_RUNNERS)}")
        return

    print("\n" + "-" * 72)
    print(f"LLM CRITIQUE — running {len(runners)} model(s) in parallel")
    print("-" * 72)

    def _invoke(name, fn):
        # Per-runner model override for gateways with multiple model choices.
        if name == "opencode" and args.opencode_model:
            return fn(state, model=args.opencode_model)
        if name == "deepseek" and args.deepseek_model:
            return fn(state, model=args.deepseek_model)
        return fn(state)

    results = []
    with ThreadPoolExecutor(max_workers=len(runners)) as ex:
        futs = {ex.submit(_invoke, name, fn): name for name, fn in runners}
        for fut in as_completed(futs):
            try:
                label, text = fut.result()
            except Exception as e:
                label, text = futs[fut], f"[crashed: {type(e).__name__}: {e}]"
            results.append((label, text))

    for label, text in results:
        print(f"\n### {label}\n")
        print(text)
        print()


if __name__ == "__main__":
    main()
