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
from .names import exact_variants, ql_escape, server_regex

log = logging.getLogger("catmap.overpass")

from .parsing import BUSINESS_KEYS  # noqa: E402,F401  (contrôle « établissement » côté Python)


# --------------------------------------------------------------------------
# Construction de la requête
# --------------------------------------------------------------------------
_ISO2 = re.compile(r"^[A-Z]{2}$")


def build_query(q: NameQuery, options: SearchOptions, settings: Settings) -> str:
    """Construit la requête Overpass QL.

    Mode « regex » (défaut, même logique que la v1 qui fonctionnait) : ``name`` (et ``brand`` si coché) en regex
    insensible à la casse (trouve « IKEA Paris Nord »), ``brand:wikidata`` en égalité exacte. Mode « exact » (rapide) : tout en égalité exacte, rappel plus faible.

    Les filtres serveur sont volontairement minimaux (comme la v1 : ni highway, ni place, ni
    junction) : « établissement » et « boundary » sont contrôlés côté Python (``parsing``),
    une regex sur les clés étant très coûteuse pour Overpass.
    """
    fast = options.search_mode == "exact"
    timeout = settings.fast_timeout_s if fast else settings.server_timeout_s
    lines = [f"[out:json][timeout:{int(timeout)}]"
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

    filters = '[!"highway"][!"place"][!"junction"]'
    elements = ["node", "way"] + (["rel[type=multipolygon]"] if options.include_relations else [])
    exact = [ql_escape(v) for v in exact_variants(q.search_name)]
    rx = server_regex(q.search_name, whole_word=options.strict_word_match)

    def conds(field: str) -> list[str]:
        """Conditions sur un tag : regex insensible à la casse (défaut) ou égalités exactes (mode rapide)."""
        if fast:
            return [f'["{field}"="{v}"]' for v in exact]
        return [f'["{field}"~"{rx}",i]']

    fields = (["name"] if options.search_name else []) + (["brand"] if options.search_brand else []) \
        + (["operator"] if options.include_operator else [])
    if not fields and not q.wikidata:
        raise ValueError("Cochez au moins un champ de recherche (name ou brand) ou renseignez un identifiant Wikidata.")

    stmts: list[str] = []
    for el in elements:
        for f in fields:
            for c in conds(f):
                stmts.append(f"  {el}{c}{filters}{scope};")
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
# Réessayer a un sens pour une surcharge passagère ; un 504 / timeout sur une requête lourde
# échouerait de nouveau : on passe directement au serveur suivant.
_RETRY_STATUS = {429, 502, 503}
_SKIP_STATUS = {500, 504}


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

    def _emit(self, on_event, msg: str) -> None:
        log.info("%s", msg)
        if on_event:
            try:
                on_event(msg)
            except Exception:  # un affichage ne doit jamais casser la recherche
                pass

    def fetch(self, query: str, use_cache: bool = True,
              on_event: Optional[Callable[[str], None]] = None) -> FetchResult:
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

        budget = self.settings.total_budget_s
        attempts = 0
        partial: Optional[OverpassPartialResult] = None
        errors: list[str] = []

        def remaining() -> float:
            return budget - (time.monotonic() - t0) if budget > 0 else float("inf")

        def fail(endpoint: str, msg: str) -> None:
            errors.append(f"{endpoint} : {msg}")
            log.warning("%s : %s", endpoint, msg)

        for endpoint in self.settings.overpass_urls:
            for attempt in range(1, self.settings.max_attempts + 1):
                if remaining() <= 0:
                    errors.append(f"budget de temps épuisé ({budget:.0f} s)")
                    break
                attempts += 1
                self._throttle()
                self._emit(on_event, f"Interrogation de {endpoint} (essai {attempt}/{self.settings.max_attempts})…")
                timeout = max(5.0, min(_client_timeout(query, self.settings), remaining()))
                try:
                    resp = self.session.post(
                        endpoint, data={"data": query}, headers=self._headers(), timeout=timeout,
                    )
                except requests.Timeout as e:
                    fail(endpoint, f"délai dépassé après {timeout:.0f} s ({type(e).__name__}) ; requête trop lourde ou serveur saturé")
                    break  # inutile de rejouer la même requête sur ce serveur
                except requests.ConnectionError as e:
                    fail(endpoint, f"connexion impossible : {str(e)[:300]}")
                    if attempt < self.settings.max_attempts and remaining() > 0:
                        self._backoff(attempt)
                        continue
                    break
                except requests.RequestException as e:
                    fail(endpoint, f"{type(e).__name__} : {str(e)[:300]}")
                    break

                status = resp.status_code
                if status == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        fail(endpoint, "réponse non JSON")
                        break
                    remark = str(data.get("remark") or "")
                    if remark and _FAILURE_REMARK.search(remark):
                        fail(endpoint, remark)
                        partial = OverpassPartialResult(remark, data, endpoint)
                        break
                    if self.settings.cache_enabled:
                        self._cache_write(key, query, endpoint, data)
                    return FetchResult(
                        data=data, endpoint=endpoint, snapshot=_snapshot(data), from_cache=False,
                        attempts=attempts, elapsed_s=time.monotonic() - t0, remark=remark,
                    )
                if status == 400:
                    raise OverpassQueryError(f"{endpoint} a refusé la requête : {resp.text[:500]}")
                fail(endpoint, f"HTTP {status}")
                if status in _RETRY_STATUS and attempt < self.settings.max_attempts and remaining() > 0:
                    self._backoff(attempt, resp.headers.get("Retry-After"))
                    continue
                break

        if partial is not None:
            raise partial
        raise OverpassUnavailable(" | ".join(errors) if errors else "aucun endpoint configuré")

    def ping(self) -> list[dict]:
        """Diagnostic : requête minimale sur chaque endpoint (latence, statut, message d'erreur)."""
        out = []
        q = "[out:json][timeout:10];node(1);out;"
        for ep in self.settings.overpass_urls:
            t = time.monotonic()
            row = {"endpoint": ep, "ok": False, "latence_s": None, "message": ""}
            try:
                r = self.session.post(ep, data={"data": q}, headers=self._headers(),
                                      timeout=self.settings.ping_timeout_s)
                row["latence_s"] = round(time.monotonic() - t, 2)
                if r.status_code == 200:
                    row["ok"] = True
                    row["message"] = ("OK " + _snapshot(r.json())).strip()
                else:
                    row["message"] = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                row["latence_s"] = round(time.monotonic() - t, 2)
                row["message"] = f"{type(e).__name__} : {str(e)[:300]}"
            except ValueError:
                row["ok"], row["message"] = False, "réponse non JSON"
            out.append(row)
        return out


def _client_timeout(query: str, settings: Settings) -> float:
    """Délai client = [timeout:] annoncé dans la requête + marge (la requête fixe sa propre limite)."""
    m = re.match(r"\[out:json\]\[timeout:(\d+)\]", query)
    return (float(m.group(1)) + 20.0) if m else settings.client_timeout_s


def _snapshot(data: dict) -> str:
    """Date de la base OSM interrogée (traçabilité / reproductibilité)."""
    return str((data.get("osm3s") or {}).get("timestamp_osm_base") or "")
