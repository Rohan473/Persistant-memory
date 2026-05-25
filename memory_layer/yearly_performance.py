"""
Parse year-by-year performance tables from alpha markdown files.
Bridges the regime_analysis.py data into the NetworkX graph.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import memory_layer.regime_analysis as regime_analysis
except ImportError:
    regime_analysis = None


@dataclass
class YearlyMetrics:
    """Parsed yearly performance metrics from an alpha."""
    year: int
    sharpe: Optional[float]
    turnover: Optional[float]
    fitness: Optional[float]
    returns: Optional[float]
    drawdown: Optional[float]
    margin: Optional[float]
    regime: str


def parse_yearly_table(markdown_body: str) -> List[YearlyMetrics]:
    """Parse the year-by-year breakdown table from alpha markdown body."""
    if not markdown_body:
        return []

    # Separator chars exclude \n so the pattern can't bleed into the first data row.
    table_pattern = (
        r'\|\s*Year\s*\|([^\n]*)\n'      # header row (capture columns after Year)
        r'\|[-|: \t]+\|[^\n]*\n'         # separator row — no \n inside the charclass
        r'((?:\|[^\n]*\n)+)'             # data rows
    )
    match = re.search(table_pattern, markdown_body, re.IGNORECASE)
    if not match:
        return []

    header_line = match.group(1)
    table_rows = match.group(2).strip().split('\n')

    # Map column-name → index (0-based across non-empty cells, including Year at 0)
    header_cells = [c.strip().lower() for c in header_line.split('|') if c.strip()]
    col_map: Dict[str, int] = {}
    for i, h in enumerate(header_cells):
        idx = i + 1  # +1 because Year occupies index 0 in the row cells
        if 'sharpe' in h:        col_map.setdefault('sharpe', idx)
        elif 'turnover' in h:    col_map.setdefault('turnover', idx)
        elif 'fitness' in h:     col_map.setdefault('fitness', idx)
        elif 'return' in h:      col_map.setdefault('returns', idx)
        elif 'drawdown' in h:    col_map.setdefault('drawdown', idx)
        elif 'margin' in h:      col_map.setdefault('margin', idx)

    def parse_float(cell: str) -> Optional[float]:
        if cell is None:
            return None
        # Unicode minus, en-dash, em-dash → ASCII hyphen
        cell = cell.replace('−', '-').replace('–', '-').replace('—', '-')
        cell = cell.replace('+', '').replace('%', '').strip()
        for char in '‱‰·•':
            cell = cell.replace(char, '')
        cell = cell.strip()
        if not cell or cell.lower() in ('none', 'n/a', 'nan', '-'):
            return None
        try:
            return float(cell)
        except ValueError:
            return None

    results = []
    for row in table_rows:
        cells = [c.strip() for c in row.split('|')]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        try:
            year = int(cells[0])
        except (ValueError, IndexError):
            continue

        def get(field: str) -> Optional[float]:
            idx = col_map.get(field)
            if idx is None or idx >= len(cells):
                return None
            return parse_float(cells[idx])

        results.append(YearlyMetrics(
            year=year,
            sharpe=get('sharpe'),
            turnover=get('turnover'),
            fitness=get('fitness'),
            returns=get('returns'),
            drawdown=get('drawdown'),
            margin=get('margin'),
            regime=_infer_regime(year),
        ))

    return results


def _infer_regime(year: int) -> str:
    """Infer market regime for a given year using regime_analysis.REGIMES."""
    if regime_analysis is None:
        return "unknown"

    for regime, info in regime_analysis.REGIMES.items():
        if year in info.get("years", []):
            return regime
    return "normal"


def parse_all_alpha_years(alpha_files) -> Dict[str, List[YearlyMetrics]]:
    """Parse yearly metrics for all alpha files."""
    import frontmatter
    from pathlib import Path

    results = {}

    for alpha_file in sorted(Path(alpha_files).glob("alpha_*.md")):
        if alpha_file.name.startswith("_"):
            continue
        try:
            post = frontmatter.load(str(alpha_file))
            alpha_id = post.metadata.get("id", alpha_file.stem)
            yearly = parse_yearly_table(post.content)
            if yearly:
                results[alpha_id] = yearly
        except Exception:
            continue

    return results