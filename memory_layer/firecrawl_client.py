"""
Firecrawl MCP Client
Handles web scraping and content extraction.
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import requests

from .firecrawl_config import config, ensure_directories

class FirecrawlClient:
    """Client for Firecrawl MCP server."""

    def __init__(self, api_url: Optional[str] = None):
        self.api_url = api_url or config.api_url
        ensure_directories()

    def is_available(self) -> bool:
        """Check if Firecrawl server is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def crawl_url(self, url: str, options: Optional[Dict] = None) -> Dict:
        """
        Crawl a URL and extract content.

        Args:
            url: URL to crawl
            options: Optional crawl options (formats, etc.)

        Returns:
            Dict with extracted content and metadata
        """
        payload = {
            "url": url,
            "options": options or {
                "formats": ["markdown", "html", "text"],
                "onlyMainContent": True,
            }
        }

        try:
            response = requests.post(
                f"{self.api_url}/crawl",
                json=payload,
                timeout=config.timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}

    def scrape_url(self, url: str, formats: List[str] = None) -> Dict:
        """
        Scrape a single URL with specified formats.

        Args:
            url: URL to scrape
            formats: List of formats (markdown, html, text, links, metadata)

        Returns:
            Dict with scraped content
        """
        if formats is None:
            formats = ["markdown", "metadata"]

        payload = {
            "url": url,
            "formats": formats
        }

        try:
            response = requests.post(
                f"{self.api_url}/scrape",
                json=payload,
                timeout=config.timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}

    def extract_from_markdown(self, markdown: str, url: str) -> Dict:
        """
        Extract structured research entities from markdown content.

        This is where the magic happens - converting raw scraped content
        into research-relevant structured data.
        """
        from .knowledge_parser import extract_research_entities

        entities = extract_research_entities(markdown, url)

        return {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "entities": entities,
            "content_length": len(markdown),
        }

    def save_raw_content(self, url: str, content: Dict) -> Path:
        """Save raw content to disk."""
        filename = self._url_to_filename(url)
        filepath = config.raw_content_dir / f"{filename}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "content": content
            }, f, indent=2, ensure_ascii=False)

        return filepath

    def save_extracted(self, url: str, extracted: Dict) -> Path:
        """Save extracted entities to disk."""
        filename = self._url_to_filename(url)
        filepath = config.extracted_dir / f"{filename}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)

        return filepath

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename."""
        import hashlib
        # Use MD5 hash for safe filename
        hash_obj = hashlib.md5(url.encode())
        return hash_obj.hexdigest()[:16]


def crawl_and_extract(url: str, save_raw: bool = True) -> Dict:
    """
    Complete crawl + extract pipeline.

    Args:
        url: URL to crawl
        save_raw: Whether to save raw content

    Returns:
        Extracted research entities
    """
    client = FirecrawlClient()

    # Check availability
    if not client.is_available():
        return {"error": "Firecrawl not available", "status": "unavailable"}

    # Scrape
    result = client.scrape_url(url, formats=["markdown", "metadata"])

    if "error" in result:
        return result

    # Get markdown content
    markdown = ""
    if "data" in result:
        markdown = result["data"].get("markdown", "")
        metadata = result["data"].get("metadata", {})

    if not markdown:
        return {"error": "No content extracted", "status": "empty"}

    # Save raw if requested
    if save_raw:
        client.save_raw_content(url, result)

    # Extract entities
    extracted = client.extract_from_markdown(markdown, url)

    # Save extracted
    client.save_extracted(url, extracted)

    return extracted