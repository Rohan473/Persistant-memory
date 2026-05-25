"""
Retrieval evaluation benchmark.
"""

import json
from pathlib import Path
from typing import List, Dict, Set
from memory_layer import retrieve_hybrid, config

# Default benchmark queries
DEFAULT_BENCHMARKS = [
    {
        "query": "fundamental value alphas failing low sharpe",
        "expected": ["alpha_0007", "alpha_0020", "alpha_0010"],
        "filters": {"concepts": ["value", "fundamental"]}
    },
    {
        "query": "mean reversion with ts_rank operator",
        "expected": ["alpha_0604", "alpha_0105", "alpha_0023"],
        "filters": {"operators": ["ts_rank"]}
    },
    {
        "query": "sentiment alphas using analyst data",
        "expected": ["alpha_0001", "alpha_0002", "alpha_0021"],
        "filters": {"concepts": ["sentiment"]}
    },
    {
        "query": "high turnover failure modes",
        "expected": ["alpha_0108", "alpha_0407", "alpha_0500"],
        "filters": {"failure_modes": ["high_turnover"]}
    },
    {
        "query": "quality concepts related to fundamental data",
        "expected": ["concept_fundamental", "concept_quality"],
        "filters": {"node_types": ["Concept"]}
    },
    {
        "query": "operators used in momentum alphas",
        "expected": ["operator_ts_decay_linear", "operator_ts_rank"],
        "filters": {"concepts": ["momentum"]}
    },
    {
        "query": "Good rated alphas with high fitness",
        "expected": [],
        "filters": {"rating": "Good"}
    },
    {
        "query": "alphas derived from alpha_0017",
        "expected": ["alpha_0020", "alpha_0016", "alpha_0017"],
        "filters": {}
    }
]

def load_benchmarks(path: Path = None) -> List[Dict]:
    """Load benchmarks from file or use defaults."""
    if path and path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return DEFAULT_BENCHMARKS

def evaluate_retrieval(query: Dict, k: int = 10) -> Dict:
    """
    Evaluate retrieval for a single query benchmark.

    Returns precision, recall, and hit list.
    """
    expected = set(query.get("expected", []))
    filters = query.get("filters", {})

    # Run retrieval
    results = retrieve_hybrid(
        query=query["query"],
        k=k,
        operators=filters.get("operators"),
        concepts=filters.get("concepts"),
        node_types=filters.get("node_types"),
    )

    # Extract retrieved IDs (normalize to lowercase for comparison)
    retrieved = set()
    for r in results:
        name = r.get("name", "").lower()
        node_id = r.get("id", "").lower()
        # Extract just the name part
        if "::" in node_id:
            retrieved.add(node_id.split("::")[1].lower())
        else:
            retrieved.add(name.lower())

    # Normalize expected
    expected_norm = set()
    for e in expected:
        expected_norm.add(e.lower())

    # Calculate metrics
    hits = retrieved.intersection(expected_norm)
    precision = len(hits) / len(retrieved) if retrieved else 0
    recall = len(hits) / len(expected_norm) if expected_norm else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "query": query["query"],
        "expected": list(expected_norm),
        "retrieved": list(retrieved),
        "hits": list(hits),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "retrieved_count": len(retrieved),
        "expected_count": len(expected_norm)
    }

def run_benchmarks(benchmarks: List[Dict] = None, k: int = 10) -> Dict:
    """Run all benchmarks and compute aggregate metrics."""
    if benchmarks is None:
        benchmarks = load_benchmarks()

    results = []
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for b in benchmarks:
        result = evaluate_retrieval(b, k=k)
        results.append(result)
        total_precision += result["precision"]
        total_recall += result["recall"]
        total_f1 += result["f1"]

    n = len(results)
    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_f1 = total_f1 / n

    return {
        "results": results,
        "summary": {
            "total_queries": n,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1
        }
    }

def print_report(eval_results: Dict) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "="*80)
    print("RETRIEVAL EVALUATION REPORT")
    print("="*80)

    summary = eval_results["summary"]
    print(f"\nTotal queries: {summary['total_queries']}")
    print(f"Average Precision: {summary['avg_precision']:.3f}")
    print(f"Average Recall:    {summary['avg_recall']:.3f}")
    print(f"Average F1:        {summary['avg_f1']:.3f}")

    print("\n" + "-"*80)
    print("Per-query results:")
    print("-"*80)

    for r in eval_results["results"]:
        print(f"\nQuery: {r['query']}")
        print(f"  Expected: {r['expected_count']} | Retrieved: {r['retrieved_count']}")
        print(f"  Hits: {r['hits']}")
        print(f"  P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")

    print("\n" + "="*80)

def save_report(eval_results: Dict, path: Path) -> None:
    """Save evaluation results to JSON."""
    with open(path, "w") as f:
        json.dump(eval_results, f, indent=2)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run retrieval benchmarks")
    parser.add_argument("--benchmarks", type=str, help="Path to benchmarks JSON file")
    parser.add_argument("-k", type=int, default=10, help="Retrieve top k results")
    parser.add_argument("--output", type=str, help="Save results to JSON")
    args = parser.parse_args()

    benchmarks_path = Path(args.benchmarks) if args.benchmarks else None
    benchmarks = load_benchmarks(benchmarks_path)

    results = run_benchmarks(benchmarks, k=args.k)
    print_report(results)

    if args.output:
        save_report(results, Path(args.output))
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()