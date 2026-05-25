"""
Migration Script: Initialize Institutional Features
Run this to migrate existing alpha data to the new institutional system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate():
    """Migrate existing data to institutional features."""
    print("=" * 60)
    print("MIGRATING TO INSTITUTIONAL RESEARCH SYSTEM")
    print("=" * 60)

    from memory_layer.structure import load_metadata
    from memory_layer.factor_ontology import classify_alpha, ontology_engine
    from memory_layer.correlation_engine import register_alpha_factors
    from memory_layer.failure_learning import analyze_failure
    from memory_layer.vector_memory import add_semantic_memory, MemoryType

    print("\n[1/5] Loading graph metadata...")
    metadata = load_metadata()
    alphas = [m for m in metadata if m.get("node_type") == "Alpha"]
    print(f"    Found {len(alphas)} alphas to migrate")

    print("\n[2/5] Classifying alphas into factor families...")
    classified_count = 0
    for alpha in alphas:
        alpha_id = alpha.get("name", "")
        expression = alpha.get("expression", "")
        datafields = alpha.get("datafields", [])
        operators = alpha.get("operators", [])
        concepts = alpha.get("concepts", [])

        exposures = classify_alpha(expression, datafields, operators, concepts)
        if exposures:
            ontology_engine.save_classifications(alpha_id, exposures)
            classified_count += 1

    print(f"    Classified {classified_count} alphas into factor families")

    print("\n[3/5] Registering alphas for correlation analysis...")
    registered_count = 0
    for alpha in alphas:
        alpha_id = alpha.get("name", "")
        expression = alpha.get("expression", "")
        datafields = alpha.get("datafields", [])
        operators = alpha.get("operators", [])
        concepts = alpha.get("concepts", [])

        exposures = classify_alpha(expression, datafields, operators, concepts)
        factor_families = [e["factor_family"] for e in exposures]

        register_alpha_factors(alpha_id, factor_families, concepts, datafields)
        registered_count += 1

    print(f"    Registered {registered_count} alphas for correlation analysis")

    print("\n[4/5] Learning from existing failures...")
    failure_count = 0
    for alpha in alphas:
        alpha_id = alpha.get("name", "")
        expression = alpha.get("expression", "")
        datafields = alpha.get("datafields", [])
        operators = alpha.get("operators", [])
        concepts = alpha.get("concepts", [])
        failure_modes = alpha.get("failure_modes", [])
        sharpe = alpha.get("sharpe")
        turnover = alpha.get("turnover")
        fitness = alpha.get("fitness")
        correlated_with = alpha.get("correlated_with", [])
        neutralization = alpha.get("neutralization", "market")

        if failure_modes:
            analyze_failure(
                alpha_id, expression, datafields, operators, concepts,
                failure_modes, sharpe, turnover, fitness, correlated_with, neutralization
            )
            failure_count += 1

    print(f"    Analyzed {failure_count} failed alphas")

    print("\n[5/5] Creating semantic memories for top alphas...")
    memory_count = 0
    top_alphas = sorted(
        [a for a in alphas if a.get("sharpe")],
        key=lambda x: x.get("sharpe", 0),
        reverse=True
    )[:20]

    for alpha in top_alphas:
        content = f"Alpha {alpha['name']}: {alpha.get('expression', '')}"
        add_semantic_memory(
            content,
            MemoryType.ALPHA_EXPRESSION,
            metadata={"sharpe": alpha.get("sharpe")},
            tags=[alpha.get("rating", "")]
        )
        memory_count += 1

    print(f"    Created {memory_count} semantic memories")

    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)

    from memory_layer import get_stats
    stats = get_stats()
    print(f"\nSystem Stats:")
    print(f"  - Vector store points: {stats.get('points_count', 0)}")
    print(f"  - Alphas classified: {classified_count}")
    print(f"  - Alphas registered: {registered_count}")
    print(f"  - Failure patterns learned: {failure_count}")
    print(f"  - Semantic memories: {memory_count}")

    from memory_layer.vector_memory import get_vector_memory_stats
    vm_stats = get_vector_memory_stats()
    print(f"\nVector Memory Stats:")
    print(f"  - Total memories: {vm_stats['total_memories']}")
    print(f"  - By type: {vm_stats['by_type']}")

    print("\nTo access new features, use the API endpoints:")
    print("  - POST /factor/classify - Classify alpha into factor families")
    print("  - GET /regime/list - List market regimes")
    print("  - POST /correlation/register - Register alpha for correlation")
    print("  - POST /query/nl - Natural language queries")
    print("  - POST /copilot/* - Research copilot features")

    return {
        "classified": classified_count,
        "registered": registered_count,
        "failures_analyzed": failure_count,
        "memories_created": memory_count
    }


if __name__ == "__main__":
    migrate()