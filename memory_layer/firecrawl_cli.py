"""
Firecrawl CLI - Web research ingestion for memory layer.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_layer.firecrawl_client import FirecrawlClient, crawl_and_extract
from memory_layer.firecrawl_config import config, get_firecrawl_api_key
from memory_layer.knowledge_parser import merge_with_internal_memory


def cmd_crawl(args):
    """Crawl a URL and extract research entities."""
    print(f"Crawling: {args.url}")

    result = crawl_and_extract(args.url, save_raw=args.save_raw)

    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    # Merge with internal memory
    if args.merge:
        result = merge_with_internal_memory(result)

    # Print summary
    print("\n--- Extraction Summary ---")
    meta = result.get("metadata", {})
    print(f"Source type: {meta.get('source_type', 'unknown')}")
    print(f"Concepts found: {meta.get('concept_count', 0)}")
    print(f"Operators found: {meta.get('operator_count', 0)}")
    print(f"Formulas found: {meta.get('formula_count', 0)}")

    if args.merge:
        links = result.get("internal_links", {})
        print(f"\n--- Internal Memory Links ---")
        print(f"Concepts matched: {links.get('concepts_matched', [])}")
        print(f"New concepts: {links.get('new_concepts', [])}")
        print(f"Operators matched: {links.get('operators_matched', [])}")

    if args.show_content:
        print("\n--- Extracted Entities ---")
        print(f"Concepts: {[c['concept'] for c in result.get('concepts', [])[:5]]}")
        print(f"Operators: {[o['operator'] for o in result.get('operators', [])[:5]]}")

    print(f"\nSaved to: {config.extracted_dir}")
    return 0


def cmd_check(args):
    """Check if Firecrawl server is available."""
    client = FirecrawlClient()

    if client.is_available():
        print("✅ Firecrawl MCP server is running")
        return 0
    else:
        print("❌ Firecrawl MCP server is not available")
        print(f"   Start with: python -m memory_layer.firecrawl_cli docker")
        return 1


def cmd_docker(args):
    """Start Firecrawl MCP Docker container."""
    import subprocess

    print("Starting Firecrawl MCP Docker...")

    api_key = get_firecrawl_api_key()
    if not api_key:
        print("❌ FIRECRAWL_API_KEY environment variable not set")
        print("   Set it with: $env:FIRECRAWL_API_KEY='your-api-key' (PowerShell)")
        print("   Or: export FIRECRAWL_API_KEY=your-api-key (bash)")
        print("   Get key from: https://www.firecrawl.dev/app/api-keys")
        return 1

    # Check if docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except:
        print("❌ Docker not available")
        return 1

    # Check if container already running
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={config.container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if config.container_name in result.stdout:
        print(f"✅ Container '{config.container_name}' already running")
        return 0

    # Check if container exists but stopped
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={config.container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if config.container_name in result.stdout:
        print(f"Starting existing container...")
        subprocess.run(["docker", "start", config.container_name], check=True)
    else:
        print(f"Creating new container...")
        cmd = [
            "docker", "run", "-d",
            "--name", config.container_name,
            "-p", f"{config.port}:3001",
            "-e", f"FIRECRAWL_API_KEY={api_key}",
            config.docker_image
        ]
        subprocess.run(cmd, check=True)

    print(f"✅ Firecrawl MCP running on http://localhost:{config.port}")
    return 0


def cmd_list(args):
    """List previously extracted content."""
    from datetime import datetime

    extracted_dir = config.extracted_dir

    if not extracted_dir.exists():
        print("No extracted content found")
        return 0

    files = sorted(extracted_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not files:
        print("No extracted content found")
        return 0

    print(f"Found {len(files)} extracted documents:\n")

    for f in files[:args.limit]:
        with open(f) as fp:
            data = json.load(fp)
            meta = data.get("metadata", {})
            concepts = data.get("concepts", [])
            print(f"  {f.name}")
            print(f"    URL: {meta.get('source_url', 'N/A')[:60]}...")
            print(f"    Type: {meta.get('source_type', 'unknown')}")
            print(f"    Concepts: {[c['concept'] for c in concepts[:3]]}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Firecrawl Web Research Ingestion")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Check
    check_parser = subparsers.add_parser("check", help="Check if Firecrawl is running")
    check_parser.set_defaults(func=cmd_check)

    # Docker
    docker_parser = subparsers.add_parser("docker", help="Start Firecrawl Docker container")
    docker_parser.set_defaults(func=cmd_docker)

    # Crawl
    crawl_parser = subparsers.add_parser("crawl", help="Crawl URL and extract research entities")
    crawl_parser.add_argument("url", type=str, help="URL to crawl")
    crawl_parser.add_argument("--save-raw", action="store_true", default=True, help="Save raw content")
    crawl_parser.add_argument("--merge", action="store_true", default=True, help="Merge with internal memory")
    crawl_parser.add_argument("--show-content", action="store_true", help="Show extracted content")
    crawl_parser.set_defaults(func=cmd_crawl)

    # List
    list_parser = subparsers.add_parser("list", help="List previously extracted content")
    list_parser.add_argument("--limit", type=int, default=10, help="Number to show")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
