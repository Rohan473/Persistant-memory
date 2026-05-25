"""
Command-line interface for memory layer.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from memory_layer import (
    config,
    extract_all_metadata,
    save_metadata,
    load_metadata,
    ingest_all,
    ingest_node_type,
    retrieve_hybrid,
    format_context,
    compile_context,
    get_stats,
    version
)
from memory_layer.structure import load_metadata as load_meta


def cmd_ingest(args):
    """Ingest nodes into memory."""
    if args.all:
        result = ingest_all(force=args.force, create_version=not args.no_version)
        print(f"Ingest complete:")
        print(f"  Ingested: {result['ingested']}")
        print(f"  Skipped: {result['skipped']}")
        print(f"  Total: {result['total']}")
    elif args.node_type:
        result = ingest_node_type(args.node_type, force=args.force)
        print(f"Ingested {result.get('ingested', 0)} {args.node_type} nodes")
    else:
        print("Error: Specify --all or --node-type")
        sys.exit(1)


def cmd_search(args):
    """Search memories."""
    results = retrieve_hybrid(
        query=args.query,
        k=args.k,
        operators=args.operators.split(",") if args.operators else None,
        concepts=args.concepts.split(",") if args.concepts else None,
        node_types=args.types.split(",") if args.types else None,
    )

    print(f"\nFound {len(results)} results:\n")

    for i, r in enumerate(results[:args.k]):
        print(f"{i+1}. {r['name']} ({r['node_type']})")
        print(f"   Score: {r['score']:.3f} | Retrieval: {r['retrieval_score']:.3f}")
        print(f"   Summary: {r['structured_summary'][:100]}...")
        print()


def cmd_context(args):
    """Get formatted context for LLM."""
    results = retrieve_hybrid(
        query=args.query,
        k=args.k,
        operators=args.operators.split(",") if args.operators else None,
        concepts=args.concepts.split(",") if args.concepts else None,
    )

    context = format_context(
        results,
        include_structured=args.structured,
        include_raw=args.raw,
        max_memories=args.k
    )

    print(context)

    if args.budget:
        from memory_layer.budget import compile_context, print_budget_report
        print("\n--- Budget Report ---")
        print_budget_report(args.query, results, config.context_budget_tokens)


def cmd_stats(args):
    """Show memory statistics."""
    stats = get_stats()
    print(f"Qdrant Collection: {config.collection_name}")
    print(f"Points: {stats.get('points_count', 'N/A')}")
    print(f"Vectors: {stats.get('vectors_count', 'N/A')}")

    # Also show metadata stats
    if config.metadata_output.exists():
        metadata = load_meta()
        by_type = {}
        for m in metadata:
            t = m.get("node_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        print("\nMetadata breakdown:")
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")


def cmd_metadata(args):
    """Show extracted metadata."""
    if args.regenerate:
        metadata = extract_all_metadata()
        save_metadata(metadata)
        print(f"Extracted {len(metadata)} nodes")
    else:
        metadata = load_meta()
        print(f"Loaded {len(metadata)} nodes")

    if args.node_type:
        filtered = [m for m in metadata if m.get("node_type") == args.node_type]
        print(f"\n{args.node_type}: {len(filtered)} nodes")
        for m in filtered[:args.limit]:
            print(f"  - {m['name']}: {m.get('structured_summary', '')[:80]}")
    else:
        by_type = {}
        for m in metadata:
            t = m.get("node_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        print("\nBy type:")
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")


def cmd_versions(args):
    """List version snapshots."""
    snapshots = version.list_snapshots()
    if not snapshots:
        print("No snapshots found")
        return

    print(f"Found {len(snapshots)} snapshots:\n")
    for s in snapshots:
        print(f"{s['created_at']} | {s['filename']}")
        print(f"  Nodes: {s['node_count']} | Label: {s.get('label', 'N/A')}")
        print()


def main():
    parser = argparse.ArgumentParser(description="WQ Brain Memory Layer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest graph into memory")
    ingest_parser.add_argument("--all", action="store_true", help="Ingest all node types")
    ingest_parser.add_argument("--node-type", type=str, help="Ingest specific node type")
    ingest_parser.add_argument("--force", action="store_true", help="Force re-ingest all")
    ingest_parser.add_argument("--no-version", action="store_true", help="Skip version snapshot")
    ingest_parser.set_defaults(func=cmd_ingest)

    # Search
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("-k", "--limit", type=int, default=5, dest="k")
    search_parser.add_argument("--operators", type=str, help="Filter by operators (comma-separated)")
    search_parser.add_argument("--concepts", type=str, help="Filter by concepts (comma-separated)")
    search_parser.add_argument("--types", type=str, help="Filter by node types (comma-separated)")
    search_parser.set_defaults(func=cmd_search)

    # Context
    context_parser = subparsers.add_parser("context", help="Get LLM-ready context")
    context_parser.add_argument("query", type=str, help="Search query")
    context_parser.add_argument("-k", "--limit", type=int, default=5, dest="k")
    context_parser.add_argument("--operators", type=str)
    context_parser.add_argument("--concepts", type=str)
    context_parser.add_argument("--structured", action="store_true", default=True, help="Include structured info")
    context_parser.add_argument("--raw", action="store_true", help="Include raw graph data")
    context_parser.add_argument("--budget", action="store_true", help="Show budget report")
    context_parser.set_defaults(func=cmd_context)

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Metadata
    meta_parser = subparsers.add_parser("metadata", help="Show extracted metadata")
    meta_parser.add_argument("--regenerate", action="store_true", help="Regenerate from graph")
    meta_parser.add_argument("--node-type", type=str, help="Filter by type")
    meta_parser.add_argument("--limit", type=int, default=10, help="Limit results")
    meta_parser.set_defaults(func=cmd_metadata)

    # Versions
    ver_parser = subparsers.add_parser("versions", help="List version snapshots")
    ver_parser.set_defaults(func=cmd_versions)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()