"""
WQ Brain catalogue cache.

Fetches the full operator + datafield catalogue from the platform API and
persists it to memory_layer/brain_catalogue.json so the graph build can
treat "available but unused" as a first-class concept.

Datafields are per-(region, delay, universe). The catalogue file stores
each (region, delay, universe) snapshot under its own key so users with
multiple settings can keep them all on disk.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CATALOGUE_PATH = Path(__file__).parent / "brain_catalogue.json"


@dataclass
class CatalogueSnapshot:
    fetched_at: str
    operators: List[Dict[str, Any]] = field(default_factory=list)
    # key: f"{region}|{delay}|{universe}" → list[datafield dict]
    datafields: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = CATALOGUE_PATH) -> "CatalogueSnapshot":
        if not path.exists():
            return cls(fetched_at="")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            fetched_at=data.get("fetched_at", ""),
            operators=data.get("operators", []),
            datafields=data.get("datafields", {}),
        )

    def save(self, path: Path = CATALOGUE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fetched_at": self.fetched_at,
                    "operators": self.operators,
                    "datafields": self.datafields,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def settings_key(self, region: str, delay: int, universe: str) -> str:
        return f"{region}|{delay}|{universe}"

    def datafields_for(self, region: str, delay: int, universe: str) -> List[Dict[str, Any]]:
        return self.datafields.get(self.settings_key(region, delay, universe), [])

    def all_datafield_ids(self) -> set:
        out = set()
        for lst in self.datafields.values():
            for df in lst:
                if "id" in df:
                    out.add(df["id"])
        return out


# ── frequency inference ──────────────────────────────────────────────────────
# Maps dataset.id substring → assumed update frequency. Used to refuse
# ts_decay_linear on quarterly fields and similar guards.
DATASET_FREQUENCY = {
    # quarterly / fundamental
    "fundamental": "quarterly",
    "fnd": "quarterly",
    "fn_": "quarterly",
    "analyst": "quarterly",
    "estimat": "quarterly",
    # daily
    "pv": "daily",
    "price": "daily",
    "volume": "daily",
    "model": "daily",
    "mdl": "daily",
    "social": "daily",
    "scl": "daily",
    "news": "daily",
    "option": "daily",      # options/volatility data updates intraday
    "sentiment": "daily",
    "ravenpack": "daily",
    "univ": "daily",
    # annual
    "annual": "annual",
}


def infer_frequency(dataset_id: str) -> str:
    if not dataset_id:
        return "unknown"
    d = dataset_id.lower()
    for key, freq in DATASET_FREQUENCY.items():
        if key in d:
            return freq
    return "unknown"


# ── fetchers ─────────────────────────────────────────────────────────────────

def fetch_operators(client) -> List[Dict[str, Any]]:
    return client.list_operators()


def fetch_datafields(
    client,
    *,
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    page_size: int = 50,
    progress: bool = False,
    sleep_between_pages: float = 0.0,
) -> List[Dict[str, Any]]:
    """Walk every datafield page for a (region, delay, universe) combo."""
    rows: List[Dict[str, Any]] = []
    first_page = client.list_datafields_page(
        region=region, delay=delay, universe=universe,
        limit=page_size, offset=0,
    )
    total = first_page.get("count") if isinstance(first_page, dict) else None
    rows.extend(first_page.get("results", []) if isinstance(first_page, dict) else first_page)

    if progress and total:
        print(f"  {len(rows)}/{total}", end="\r", flush=True)

    while total is None or len(rows) < total:
        if sleep_between_pages:
            time.sleep(sleep_between_pages)
        page = client.list_datafields_page(
            region=region, delay=delay, universe=universe,
            limit=page_size, offset=len(rows),
        )
        new_rows = page.get("results", []) if isinstance(page, dict) else page
        if not new_rows:
            break
        rows.extend(new_rows)
        if progress and total:
            print(f"  {len(rows)}/{total}", end="\r", flush=True)
        if total is not None and len(rows) >= total:
            break

    if progress:
        print()
    return rows


def refresh(
    client,
    *,
    settings: Optional[List[Tuple[str, int, str]]] = None,
    skip_operators: bool = False,
    skip_datafields: bool = False,
    page_size: int = 50,
    progress: bool = True,
    sleep_between_pages: float = 0.0,
) -> CatalogueSnapshot:
    """
    Refresh the catalogue cache.

    settings: list of (region, delay, universe). Defaults to
              [("USA", 1, "TOP3000"), ("USA", 1, "TOP1000"), ("USA", 1, "TOP500")].
    """
    snap = CatalogueSnapshot.load()
    snap.fetched_at = datetime.utcnow().isoformat()

    if not skip_operators:
        if progress:
            print("Fetching operators ...")
        snap.operators = fetch_operators(client)
        if progress:
            print(f"  {len(snap.operators)} operators")

    if not skip_datafields:
        if settings is None:
            settings = [("USA", 1, "TOP3000"), ("USA", 1, "TOP1000"), ("USA", 1, "TOP500")]
        for region, delay, universe in settings:
            key = snap.settings_key(region, delay, universe)
            if progress:
                print(f"Fetching datafields for {key} ...")
            rows = fetch_datafields(
                client, region=region, delay=delay, universe=universe,
                page_size=page_size, progress=progress,
                sleep_between_pages=sleep_between_pages,
            )
            snap.datafields[key] = rows

    snap.save()
    return snap


# ── analysis helpers used by query.py ────────────────────────────────────────

def datafields_unused_by_alphas(
    snap: CatalogueSnapshot,
    used_ids: Iterable[str],
    *,
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    min_coverage: float = 0.5,
    min_user_count: int = 0,
) -> List[Dict[str, Any]]:
    used = {u.lower() for u in used_ids}
    out = []
    for df in snap.datafields_for(region, delay, universe):
        if df.get("id", "").lower() in used:
            continue
        if (df.get("coverage") or 0) < min_coverage:
            continue
        if (df.get("userCount") or 0) < min_user_count:
            continue
        out.append(df)
    out.sort(key=lambda d: (d.get("userCount", 0), d.get("coverage", 0)), reverse=True)
    return out


def operators_unused_by_alphas(
    snap: CatalogueSnapshot,
    used_names: Iterable[str],
) -> List[Dict[str, Any]]:
    used = {u.lower() for u in used_names}
    return [op for op in snap.operators if op.get("name", "").lower() not in used]
