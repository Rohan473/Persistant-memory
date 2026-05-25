"""
WQ Brain API client — session-based authentication with token caching.

Endpoints assumed (VERIFY against the official WQ Brain API docs you have access to;
adjust the constants below if any path differs):

  POST {base}/authentication          → log in via HTTP Basic Auth, returns Set-Cookie
  GET  {base}/users/self              → identity check (session validity)
  GET  {base}/data-fields             → paginated datafield list
  GET  {base}/operators               → operator list
  POST {base}/simulations             → submit a simulation (returns Location header)
  GET  {base}/simulations/{id}        → poll simulation status / fetch result

The session cookie is stored at ~/.wqbrain/session.json (outside this repo).
Cookie name and TTL vary by deployment — the client auto-detects the first cookie
returned by /authentication so it's robust to renaming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import requests


DEFAULT_BASE_URL = "https://api.worldquantbrain.com"
SESSION_FILE = Path.home() / ".wqbrain" / "session.json"


class BrainAuthError(Exception):
    """Raised when authentication fails or a saved session is invalid."""


class BrainRateLimit(Exception):
    """Raised when the API returns 429."""


@dataclass
class BrainSession:
    cookie_name: str
    cookie_value: str
    base_url: str
    saved_at: str
    extra_cookies: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BrainSession":
        return cls(
            cookie_name=d["cookie_name"],
            cookie_value=d["cookie_value"],
            base_url=d.get("base_url", DEFAULT_BASE_URL),
            saved_at=d.get("saved_at", datetime.utcnow().isoformat()),
            extra_cookies=d.get("extra_cookies", {}),
        )


class BrainAPIClient:
    """Thin REST wrapper around the WQ Brain platform API."""

    def __init__(
        self,
        session: Optional[BrainSession] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        self.base_url = (session.base_url if session else base_url).rstrip("/")
        self.session = session
        self.timeout = timeout
        self._http = requests.Session()
        if session:
            self._http.cookies.set(session.cookie_name, session.cookie_value)
            for k, v in session.extra_cookies.items():
                self._http.cookies.set(k, v)

    # ── persistence ────────────────────────────────────────────────────────

    @classmethod
    def from_disk(cls, path: Path = SESSION_FILE) -> "BrainAPIClient":
        if not path.exists():
            raise BrainAuthError(
                f"No saved session at {path}. Run `python scripts/wqbrain_login.py` first."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(session=BrainSession.from_dict(data))

    def save(self, path: Path = SESSION_FILE) -> None:
        if not self.session:
            raise ValueError("No session to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.session.to_dict(), indent=2), encoding="utf-8")

    # ── low-level request ──────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        import time as _time
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        backoff = 1.0
        for attempt in range(6):
            r = self._http.request(method, url, **kwargs)
            if r.status_code in (401, 403):
                raise BrainAuthError(
                    f"{r.status_code} on {method} {path} — session likely expired. "
                    f"Re-run scripts/wqbrain_login.py."
                )
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                _time.sleep(wait)
                backoff = min(backoff * 2, 30.0)
                continue
            r.raise_for_status()
            return r
        raise BrainRateLimit(f"Persistently rate-limited on {method} {path}")

    # ── identity ──────────────────────────────────────────────────────────

    def whoami(self) -> Dict[str, Any]:
        return self._request("GET", "/users/self").json()

    # ── catalogue ─────────────────────────────────────────────────────────

    def list_operators(self) -> list:
        r = self._request("GET", "/operators")
        data = r.json()
        return data.get("results", data) if isinstance(data, dict) else data

    def list_datafields_page(
        self,
        *,
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        dataset_id: Optional[str] = None,
        instrument_type: str = "EQUITY",
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        # WQ Brain caps limit at 50; anything higher returns 400.
        limit = min(limit, 50)
        params = {
            "region": region,
            "delay": delay,
            "universe": universe,
            "instrumentType": instrument_type,
            "limit": limit,
            "offset": offset,
        }
        if dataset_id:
            params["dataset.id"] = dataset_id
        return self._request("GET", "/data-fields", params=params).json()

    def iter_datafields(
        self,
        *,
        page_size: int = 50,
        max_pages: Optional[int] = None,
        **filters,
    ) -> Iterator[Dict[str, Any]]:
        offset = 0
        page_idx = 0
        while True:
            page = self.list_datafields_page(limit=page_size, offset=offset, **filters)
            rows = page.get("results", []) if isinstance(page, dict) else page
            if not rows:
                return
            for row in rows:
                yield row
            offset += len(rows)
            page_idx += 1
            count = page.get("count") if isinstance(page, dict) else None
            if count is not None and offset >= count:
                return
            if max_pages is not None and page_idx >= max_pages:
                return

    # ── simulation ────────────────────────────────────────────────────────

    def submit_simulation(
        self,
        expression: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a simulation. Returns the simulation id from the Location header."""
        defaults = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 0,
            "neutralization": "SUBINDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "ON",
            "language": "FASTEXPR",
            "visualization": False,
        }
        if settings:
            defaults.update(settings)
        payload = {
            "type": "REGULAR",
            "settings": defaults,
            "regular": expression,
        }
        r = self._request("POST", "/simulations", json=payload)
        loc = r.headers.get("Location", "")
        if not loc:
            raise RuntimeError("Simulation submitted but no Location header returned")
        return loc.rstrip("/").rsplit("/", 1)[-1]

    def get_simulation(self, sim_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/simulations/{sim_id}").json()


# ── login helpers ────────────────────────────────────────────────────────────

def _pick_session_cookie(jar: requests.cookies.RequestsCookieJar) -> tuple[str, str]:
    """Heuristic: pick the most likely session cookie from the jar."""
    if not jar:
        raise BrainAuthError("Login succeeded but no cookies returned")
    preferred = ("t", "session", "JSESSIONID", "AWSALB")
    for name in preferred:
        v = jar.get(name)
        if v:
            return name, v
    # Fall back to the longest cookie value (session tokens tend to be long)
    items = sorted(jar.items(), key=lambda kv: -len(kv[1] or ""))
    name, value = items[0]
    return name, value


def login_with_password(
    username: str,
    password: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 15.0,
) -> BrainSession:
    sess = requests.Session()
    r = sess.post(
        f"{base_url.rstrip('/')}/authentication",
        auth=(username, password),
        timeout=timeout,
    )
    if r.status_code == 401:
        raise BrainAuthError("Invalid credentials")
    if r.status_code == 403:
        raise BrainAuthError("Account exists but is not authorized for API access")
    r.raise_for_status()
    name, value = _pick_session_cookie(sess.cookies)
    extras = {k: v for k, v in sess.cookies.items() if k != name}
    return BrainSession(
        cookie_name=name,
        cookie_value=value,
        base_url=base_url,
        saved_at=datetime.utcnow().isoformat(),
        extra_cookies=extras,
    )


def login_with_cookie(
    cookie_value: str,
    cookie_name: str = "t",
    base_url: str = DEFAULT_BASE_URL,
) -> BrainSession:
    """Construct a session from a cookie value pasted from browser DevTools."""
    return BrainSession(
        cookie_name=cookie_name,
        cookie_value=cookie_value.strip(),
        base_url=base_url,
        saved_at=datetime.utcnow().isoformat(),
    )
