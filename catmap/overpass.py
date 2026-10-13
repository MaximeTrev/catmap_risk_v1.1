"""Client Overpass robuste.

Corrige, par rapport à la v1 :
- A4  : un HTTP 200 avec ``remark`` d'erreur (timeout, mémoire) n'est plus pris pour un
        résultat complet ; il est traité comme un échec, sans jamais être mis en cache ;
- A20 : cache disque (clé = requête + espace de noms) : une requête identique n'est jamais
        rejouée ;
- retries sur 429/502/503/504 avec attente exponentielle + jitter (respect de ``Retry-After``) ;
- endpoints configurables (instance locale ou serveurs publics) ;
- aucune adresse e-mail codée en dur (contact via ``CATMAP_CONTACT``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

from .config import Settings
from .models import NameQuery, SearchOptions
from .names import server_regex

log = logging.getLogger("catmap.overpass")

BUSINESS_KEYS = (
    "shop", "amenity", "office", "craft", "industrial", "man_made", "power",
    "healthcare", "tourism", "leisure", "landuse", "building",
)


# --------------------------------------------------------------------------
# Construction de la requête
# --------------------------------------------------------------------------
_ISO2 = re.compile(r"^[A-Z]{2}$")


def build_query(q: NameQuery, options: SearchOptions, settings: Settings) -> str:
    """Construit la requête Overpass QL.

    Choix :
    - regex serveur = sous-chaîne tolérante (voir ``names.server_regex``) ; la précision « mot
      entier » est appliquée côté Python ;
    - recherche sur ``name`` ET ``brand`` (rappel), ``operator`` en option (bruit) ;
    - exclusion des rues, lieux-dits, jonctions et limites administratives ;
    - filtre « établissement » (présence d'un tag shop/amenity/office/…) activable ;
    - ``out body geom`` : géométrie complète des ways et relations, sans la récursion ``>``.
    """
    lines = [f"[out:json][timeout:{int(settings.server_timeout_s)}]"
             + (f"[maxsize:{int(settings.maxsize_mb) * 1024 * 1024}]" if settings.maxsize_mb > 0 else "")
             + ";"]

    scope = ""
    if options.countries:
        codes = sorted({c.strip().upper() for c in options.countries})
        for c in codes:
            if not _ISO2.match(c):
                raise ValueError(f"Code pays ISO 3166-1 alpha-2 invalide : {c!r}")
        areas = "".join(f'area["ISO3166-1"="{c}"][admin_level=2];' for c in codes)
        lines.append(f"({areas})->.searchArea;")
        scope = "(area.searchArea)"
    elif options.bbox:
        s, w, n, e = (float(v) for v in options.bbox)
        if not (-90 <= s < n <= 90 and -180 <= w <= 180 and -180 <= e <= 180):
            raise ValueError(f"bbox invalide : {options.bbox}")
        scope = f"({s},{w},{n},{e})"

    filters = '[!"highway"][!"place"][!"junction"][!"boundary"]'
    if options.require_business_tag:
        filters += '[~"^(' + "|".join(BUSINESS_KEYS) + ')$"~"."]'

    elements = ["node", "way"] + (["rel[type=multipolygon]"] if options.include_relations else [])
    rx = server_regex(q.search_name)
    fields = ["name", "brand"] + (["operator"] if options.include_operator else [])

    stmts: list[str] = []
    for el in elements:
        for f in fields:
            stmts.append(f'  {el}["{f}"~"{rx}",i]{filters}{scope};')
        if q.wikidata:
            stmts.append(f'  {el}["brand:wikidata"="{q.wikidata}"]{filters}{scope};')

    lines.append("(")
    lines.extend(stmts)
    lines.append(");")
    lines.append("out body geom;")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Erreurs
# --------------------------------------------------------------------------
class OverpassError(Exception):
    """Erreur générique du client Overpass."""


class OverpassQueryError(OverpassError):
    """Requête rejetée (HTTP 400) : inutile de réessayer ou de changer de serveur."""


class OverpassUnavailable(OverpassError):
    """Aucun serveur n'a fourni de réponse exploitable."""


class OverpassPartialResult(OverpassError):
    """Le serveur a répondu mais a signalé une troncature (timeout / mémoire)."""

    def __init__(self, remark: str, data: dict, endpoint: str):
        super().__init__(remark)
        self.remark = remark
        self.data = data
        self.endpoint = endpoint


@dataclass
class FetchResult:
    data: dict
    endpoint: str
    snapshot: str
    from_cache: bool
    attempts: int
    elapsed_s: float
    remark: str = ""


_FAILURE_REMARK = re.compile(r"runtime error|timed out|out of memory|too many|rate_limited", re.I)
_RETRY_STATUS = {429, 500, 502, 503, 504}


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class OverpassClient:
    def __init__(
        self,
        settings: Settings,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.session = session or requests.Session()
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_ok = 0.0

    # ---- cache ---------------------------------------------------------
    def _cache_key(self, query: str) -> str:
        h = hashlib.sha256()
        h.update(self.settings.cache_namespace.encode("utf-8"))
        h.update(b"\n")
        h.update(query.encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return Path(self.settings.cache_dir) / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                env = json.load(fh)
        except (FileNotFoundError, ValueError, OSError):
            return None
        ttl = self.settings.cache_ttl_hours
        if ttl is not None and (time.time() - float(env.get("cached_at", 0))) > ttl * 3600:
            return None
        if not isinstance(env.get("data"), dict):
            return None
        return env

    def _cache_write(self, key: str, query: str, endpoint: str, data: dict) -> None:
        try:
            d = Path(self.settings.cache_dir)
            d.mkdir(parents=True, exist_ok=True)
            env = {"cached_at": time.time(), "endpoint": endpoint, "query": query, "data": data}
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(env, fh)
            os.replace(tmp, self._cache_path(key))  # écriture atomique
        except OSError as e:  # le cache ne doit jamais faire échouer une recherche
            log.warning("Écriture du cache impossible : %s", e)

    # ---- réseau --------------------------------------------------------
    def _headers(self) -> dict:
        return {"User-Agent": self.settings.user_agent, "Accept": "application/json"}

    def _throttle(self) -> None:
        gap = self.settings.min_interval_s
        if gap <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_ok - now
            if wait > 0:
                self._sleep(wait)
                now = time.monotonic()
            self._next_ok = now + gap

    def _backoff(self, attempt: int, retry_after: Optional[str] = None) -> None:
        wait: float
        if retry_after and retry_after.strip().isdigit():
            wait = float(retry_after)
        else:
            wait = self.settings.backoff_base_s * (2 ** (attempt - 1))
            wait += random.uniform(0, self.settings.backoff_base_s)
        wait = min(wait, self.settings.backoff_max_s)
        log.info("Nouvelle tentative dans %.1f s", wait)
        self._sleep(wait)

    def fetch(self, query: str, use_cache: bool = True) -> FetchResult:
        t0 = time.monotonic()
        key = self._cache_key(query)
        if use_cache and self.settings.cache_enabled:
            hit = self._cache_read(key)
            if hit:
                data = hit["data"]
                return FetchResult(
                    data=data, endpoint=str(hit.get("endpoint", "cache")),
                    snapshot=_snapshot(data), from_cache=True, attempts=0,
                    elapsed_s=time.monotonic() - t0,
                )

        attempts = 0
        partial: Optional[OverpassPartialResult] = None
        last_error = "aucun endpoint configuré"
        for endpoint in self.settings.overpass_urls:
            for attempt in range(1, self.settings.max_attempts + 1):
                attempts += 1
                self._throttle()
                try:
                    resp = self.session.post(
                        endpoint, data={"data": query}, headers=self._headers(),
                        timeout=self.settings.client_timeout_s,
                    )
                except (requests.Timeout, requests.ConnectionError) as e:
                    last_error = f"{endpoint} : {type(e).__name__}"
                    log.warning("%s", last_error)
                    if attempt < self.settings.max_attempts:
                        self._backoff(attempt)
                        continue
                    break
                except requests.RequestException as e:
                    last_error = f"{endpoint} : {e}"
                    log.warning("%s", last_error)
                    break

                status = resp.status_code
                if status == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        last_error = f"{endpoint} : réponse non JSON"
                        log.warning("%s", last_error)
                        break
                    remark = str(data.get("remark") or "")
                    if remark and _FAILURE_REMARK.search(remark):
                        last_error = f"{endpoint} : {remark}"
                        log.warning("Résultat partiel : %s", last_error)
                        partial = OverpassPartialResult(remark, data, endpoint)
                        break  # on retentera sur un autre serveur, pas sur celui-ci
                    if self.settings.cache_enabled:
                        self._cache_write(key, query, endpoint, data)
                    return FetchResult(
                        data=data, endpoint=endpoint, snapshot=_snapshot(data), from_cache=False,
                        attempts=attempts, elapsed_s=time.monotonic() - t0, remark=remark,
                    )
                if status == 400:
                    raise OverpassQueryError(f"{endpoint} a refusé la requête : {resp.text[:500]}")
                if status in _RETRY_STATUS:
                    last_error = f"{endpoint} : HTTP {status}"
                    log.warning("%s", last_error)
                    if attempt < self.settings.max_attempts:
                        self._backoff(attempt, resp.headers.get("Retry-After"))
                        continue
                    break
                last_error = f"{endpoint} : HTTP {status}"
                log.warning("%s", last_error)
                break

        if partial is not None:
            raise partial
        raise OverpassUnavailable(last_error)


def _snapshot(data: dict) -> str:
    """Date de la base OSM interrogée (traçabilité / reproductibilité)."""
    return str((data.get("osm3s") or {}).get("timestamp_osm_base") or "")
