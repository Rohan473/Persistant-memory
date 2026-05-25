"""
Firecrawl MCP Configuration
Web scraping and content extraction for external research.
"""

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class FirecrawlConfig:
    # Docker configuration
    docker_image: str = "dhi.io/firecrawl-mcp"
    container_name: str = "firecrawl-mcp"
    port: int = 3001

    # Output directories
    raw_content_dir: Path = Path(__file__).parent.parent / "external_research" / "raw"
    extracted_dir: Path = Path(__file__).parent.parent / "external_research" / "extracted"

    # API settings
    api_url: str = "http://localhost:3001"
    timeout_seconds: int = 60

    # Extraction settings
    max_content_length: int = 50000  # Truncate very long content

    # Rate limiting
    requests_per_minute: int = 10

config = FirecrawlConfig()

def ensure_directories():
    """Create necessary directories."""
    config.raw_content_dir.mkdir(parents=True, exist_ok=True)
    config.extracted_dir.mkdir(parents=True, exist_ok=True)

def get_api_url() -> str:
    return config.api_url

def get_firecrawl_api_key() -> str:
    """Get Firecrawl API key from environment."""
    return os.getenv("FIRECRAWL_API_KEY", "")