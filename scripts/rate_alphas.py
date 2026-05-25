"""
Compute and write a `rating` field into every alpha markdown file.

Rating tiers
------------
  Good              — submitted AND Fitness >= 1.3 AND Sharpe >= 1.0
  Average           — submitted/accepted but below Good thresholds
  Needs Improvement — rejected | idea_only | iterating with failing metrics

Run:  python scripts/rate_alphas.py
"""

from pathlib import Path
from collections import Counter
import frontmatter
import yaml

BASE       = Path(__file__).resolve().parent.parent
ALPHAS_DIR = BASE / "nodes" / "alphas"


def safe_float(val):
    try:
        f = float(val)
        return None if str(val).strip() == "null" else f
    except (TypeError, ValueError):
        return None


def compute_rating(meta: dict) -> str:
    status  = str(meta.get("status") or "").strip()
    sharpe  = safe_float(meta.get("sharpe"))
    fitness = safe_float(meta.get("fitness"))

    # ── Not accepted ──────────────────────────────────────────────────────────
    if status in ("rejected", "idea_only"):
        return "Needs Improvement"

    # Iterating: only Average if it's already clearing Fitness
    if status == "iterating":
        if fitness is not None and fitness >= 1.0:
            return "Average"
        return "Needs Improvement"

    # ── Accepted / submitted ──────────────────────────────────────────────────
    if status == "submitted":
        # Good: clears both Fitness and Sharpe thresholds
        if (fitness is not None and fitness >= 1.3
                and sharpe is not None and sharpe >= 1.0):
            return "Good"
        # Average: accepted but one metric is below the bar
        return "Average"

    # Fallback
    return "Needs Improvement"


def rewrite_rating(path: Path, rating: str) -> bool:
    """Insert/update the `rating:` key in YAML frontmatter. Returns True if changed."""
    post = frontmatter.load(str(path))
    if post.metadata.get("rating") == rating:
        return False                        # already correct, skip write

    post.metadata["rating"] = rating

    # Serialise manually to preserve field order and avoid frontmatter lib quirks
    meta_lines = []
    # Write all existing keys first, replacing/inserting rating after status
    keys = list(post.metadata.keys())
    if "rating" not in keys:
        # insert after "status"
        idx = keys.index("status") + 1 if "status" in keys else len(keys)
        keys.insert(idx, "rating")

    for k in keys:
        v = post.metadata.get(k)
        if isinstance(v, list):
            if v:
                meta_lines.append(f"{k}: [{', '.join(str(i) for i in v)}]")
            else:
                meta_lines.append(f"{k}: []")
        elif v is None or v == "null":
            meta_lines.append(f"{k}: null")
        elif isinstance(v, str):
            # quote strings that need it
            if any(c in v for c in (':', '#', '[', ']', '{', '}', '*', '&', '!', '|', '>', '"')):
                escaped = v.replace('"', '\\"')
                meta_lines.append(f'{k}: "{escaped}"')
            else:
                meta_lines.append(f"{k}: {v}")
        else:
            meta_lines.append(f"{k}: {v}")

    front = "---\n" + "\n".join(meta_lines) + "\n---\n"
    body  = post.content.strip()
    path.write_text(front + ("\n\n" + body if body else ""), encoding="utf-8")
    return True


def main():
    alpha_files = sorted(ALPHAS_DIR.glob("alpha_*.md"))
    print(f"Rating {len(alpha_files)} alphas...\n")

    counts   = Counter()
    changes  = 0

    for af in alpha_files:
        try:
            post   = frontmatter.load(str(af))
            rating = compute_rating(post.metadata)
            changed = rewrite_rating(af, rating)
            counts[rating] += 1
            if changed:
                changes += 1
        except Exception as e:
            print(f"  WARN {af.name}: {e}")

    print("Rating distribution:")
    print(f"  {'Good':<22} {counts['Good']:>3}")
    print(f"  {'Average':<22} {counts['Average']:>3}")
    print(f"  {'Needs Improvement':<22} {counts['Needs Improvement']:>3}")
    print(f"\nFiles updated: {changes} / {len(alpha_files)}")

    # Print the Good ones
    print("\n--- Good alphas ---")
    for af in sorted(ALPHAS_DIR.glob("alpha_*.md")):
        post = frontmatter.load(str(af))
        if post.metadata.get("rating") == "Good":
            s = post.metadata.get("sharpe")
            f = post.metadata.get("fitness")
            expr = str(post.metadata.get("expression") or "")[:65]
            print(f"  {af.stem}  Sharpe={s}  Fitness={f}  {expr}")

    print("\n--- Average alphas ---")
    for af in sorted(ALPHAS_DIR.glob("alpha_*.md")):
        post = frontmatter.load(str(af))
        if post.metadata.get("rating") == "Average":
            s = post.metadata.get("sharpe")
            f = post.metadata.get("fitness")
            expr = str(post.metadata.get("expression") or "")[:65]
            print(f"  {af.stem}  Sharpe={s}  Fitness={f}  {expr}")


if __name__ == "__main__":
    main()
