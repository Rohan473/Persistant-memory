"""
Context budgeting: compile memories within token budget for LLM injection.
"""

from typing import List, Dict, Tuple
import re

# Rough token estimation: ~4 characters per token
CHARS_PER_TOKEN = 4

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Rough approximation: len(text) / 4
    """
    return len(text) // CHARS_PER_TOKEN

def estimate_tokens_precise(text: str) -> int:
    """
    More precise token estimation using regex to count words and punctuation.
    This is still an approximation - actual tokenizers vary.
    """
    # Count tokens more accurately: split on whitespace + punctuation
    # This approximates how subword tokenizers work
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return len(tokens)

def compress_memory(memory: Dict) -> str:
    """
    Compress a single memory into a minimal structured string.

    Format:
    [TYPE] NAME | OPS:op1,op2 | CONCEPTS:c1,c2 | FAIL:f1 | RATING:r
    """
    node_type = memory.get("node_type", "")
    name = memory.get("name", "")
    parts = [f"[{node_type}] {name}"]

    if node_type == "Alpha":
        ops = memory.get("operators", [])[:3]
        concepts = memory.get("concepts", [])[:3]
        failures = memory.get("failure_modes", [])[:2]
        rating = memory.get("rating", "")

        if ops:
            parts.append(f"OPS:{','.join(ops)}")
        if concepts:
            parts.append(f"CONCEPTS:{','.join(concepts)}")
        if failures:
            parts.append(f"FAIL:{','.join(failures)}")
        if rating:
            parts.append(f"RATING:{rating}")

    elif node_type == "Concept":
        related = memory.get("related_alphas", [])[:3]
        if related:
            parts.append(f"ALPHAS:{','.join(related)}")

    elif node_type == "FailureMode":
        failed = memory.get("failed_alphas", [])[:3]
        if failed:
            parts.append(f"FAILED:{','.join(failed)}")

    return " | ".join(parts)

def compile_context(
    memories: List[Dict],
    budget_tokens: int = 2000,
    max_memories: int = 10,
    include_metadata: bool = True
) -> Tuple[str, int, List[Dict]]:
    """
    Compile memories into context within token budget.

    Args:
        memories: List of retrieved memory records
        budget_tokens: Maximum tokens allowed
        max_memories: Maximum memories to include
        include_metadata: Include salience/importance info

    Returns:
        Tuple of (compiled_context, token_count, used_memories)
    """
    if not memories:
        return "No relevant memories found.", 0, []

    # Sort by retrieval score
    sorted_memories = sorted(
        memories,
        key=lambda m: m.get("retrieval_score", m.get("score", 0)),
        reverse=True
    )

    # Try including memories one by one until budget exceeded
    used = []
    context_parts = ["## Research Memory Context\n"]

    for mem in sorted_memories[:max_memories]:
        # Compress the memory
        compressed = compress_memory(mem)

        # Add metadata if requested
        if include_metadata:
            salience = mem.get("salience", {})
            importance = salience.get("importance", 0.5)
            reuse = salience.get("reuse_count", 0)
            compressed += f" [imp:{importance:.1f},use:{reuse}]"

        # Check if adding this would exceed budget
        test_context = "\n".join(context_parts + [compressed]) + "\n"
        estimated = estimate_tokens_precise(test_context)

        if estimated > budget_tokens:
            break

        context_parts.append(compressed)
        used.append(mem)

    final_context = "\n".join(context_parts)
    token_count = estimate_tokens_precise(final_context)

    return final_context, token_count, used

def compile_context_compact(
    memories: List[Dict],
    budget_tokens: int = 2000,
    max_memories: int = 10
) -> Tuple[str, int]:
    """
    Even more compact context compilation.
    Returns a single-line comma-separated format.
    """
    if not memories:
        return "None", 0

    sorted_memories = sorted(
        memories,
        key=lambda m: m.get("retrieval_score", m.get("score", 0)),
        reverse=True
    )

    parts = []
    for mem in sorted_memories[:max_memories]:
        node_type = mem.get("node_type", "")
        name = mem.get("name", "")

        if node_type == "Alpha":
            ops = mem.get("operators", [])[:2]
            concepts = mem.get("concepts", [])[:2]
            rating = mem.get("rating", "")
            parts.append(f"{name}({','.join(ops[:1])},{','.join(concepts[:1])})")
        else:
            parts.append(f"{name}")

    context = " | ".join(parts)
    return context, estimate_tokens_precise(context)

def print_budget_report(
    query: str,
    memories: List[Dict],
    budget_tokens: int = 2000
) -> None:
    """Print a budget usage report for debugging."""
    context, tokens, used = compile_context(memories, budget_tokens)

    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"BUDGET: {budget_tokens} tokens")
    print(f"RETRIEVED: {len(memories)} memories")
    print(f"USED: {len(used)} memories")
    print(f"TOKEN COUNT: {tokens}")
    print(f"{'='*60}")
    print(context)
    print(f"{'='*60}")