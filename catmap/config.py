"""Configuration centralisée.

Aucun secret ni adresse personnelle dans le code (corrige A2). Les valeurs viennent,
par ordre de priorité : arguments explicites > variables d'environnement > secrets
Streamlit (``.streamlit/secrets.toml``, jamais versionné) > valeurs par défaut.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

PUBLIC_OVERPASS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)


def _split(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _to_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def _to_int(value: Optional[str], default: int) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except ValueError:
        return default


def is_local_endpoint(url: str) -> bool:
    return any(h in url for h in ("://localhost", "://127.0.0.1", "://0.0.0.0", "://overpass:", "://host.docker.internal"))


@dataclass(frozen=True)
class Settings:
    # --- Overpass -------------------------------------------------------
    overpass_urls: tuple[str, ...] = PUBLIC_OVERPASS
    contact: str = ""                       # CATMAP_CONTACT : e-mail ou URL de contact (bonne pratique Overpass)
    server_timeout_s: int = 280             # [timeout:] côté serveur, recherche regex (comme la v1)
    fast_timeout_s: int = 60                # [timeout:] en mode exact (rapide)
    total_budget_s: float = 600.0           # temps maxi par entreprise, tous serveurs confondus
    ping_timeout_s: float = 15.0            # test de connexion
    area_timeout_s: int = 180               # [timeout:] d'une recherche par périmètre
    area_max_radius_m: int = 10000          # rayon maximal d'un périmètre (évite les réponses trop lourdes)
    area_max_sites: int = 50000             # sites conservés au maximum (les plus proches du centre)
    maxsize_mb: int = 0                     # 0 = ne pas fixer [maxsize:]
    max_attempts: int = 2                   # essais par endpoint (erreurs réseau / 429 / 502 / 503 seulement)
    backoff_base_s: float = 5.0             # attente de base (exponentielle + jitter)
    backoff_max_s: float = 90.0
    min_interval_s: float = 1.0             # délai mini entre deux requêtes (0 en local)
    max_workers: int = 1                    # parallélisme (à augmenter seulement en local)
    # --- Cache disque ---------------------------------------------------
    cache_dir: Path = ROOT / ".cache" / "overpass"
    cache_enabled: bool = True
    cache_ttl_hours: Optional[float] = 168.0  # None = n'expire jamais (instantané figé)
    cache_namespace: str = ""               # ex. « fr-2026-09 » pour isoler des instantanés
    # --- Sécurité / robustesse -----------------------------------------
    max_sites_per_company: int = 50000
    # --- Météo-France ---------------------------------------------------
    mf_api_key: str = ""
    mf_client_id: str = ""
    mf_client_secret: str = ""
    mf_token_url: str = "https://public-api.meteofrance.fr/token"
    mf_base_url: str = "https://public-api.meteofrance.fr/public/DPVigilance/v1"
    departements_geojson: str = ""          # chemin ou URL d'un GeoJSON des départements

    @property
    def client_timeout_s(self) -> float:
        return self.server_timeout_s + 20

    @property
    def user_agent(self) -> str:
        return f"CatMap/3.2 (contact: {self.contact})" if self.contact else "CatMap/3.2"

    @property
    def uses_public_overpass(self) -> bool:
        return any(not is_local_endpoint(u) for u in self.overpass_urls)

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)


def load_settings(env: Optional[Mapping[str, str]] = None, secrets: Optional[Mapping[str, str]] = None) -> Settings:
    """Construit les réglages à partir de l'environnement et, en option, de ``st.secrets``."""
    env = os.environ if env is None else env
    secrets = secrets or {}

    def get(key: str, default: str = "") -> str:
        v = env.get(key)
        if v in (None, ""):
            v = secrets.get(key)
        return default if v in (None, "") else str(v)

    urls = _split(get("OVERPASS_URLS")) or PUBLIC_OVERPASS
    local = all(is_local_endpoint(u) for u in urls)

    ttl_raw = get("CATMAP_CACHE_TTL_HOURS", "168").strip().lower()
    ttl: Optional[float]
    if ttl_raw in ("never", "inf", "none", "-1"):
        ttl = None
    else:
        ttl = _to_float(ttl_raw, 168.0)

    return Settings(
        overpass_urls=urls,
        contact=get("CATMAP_CONTACT"),
        server_timeout_s=_to_int(get("OVERPASS_TIMEOUT_S"), 280),
        fast_timeout_s=_to_int(get("OVERPASS_FAST_TIMEOUT_S"), 60),
        total_budget_s=_to_float(get("OVERPASS_TOTAL_BUDGET_S"), 600.0),
        area_timeout_s=_to_int(get("OVERPASS_AREA_TIMEOUT_S"), 180),
        area_max_radius_m=max(50, _to_int(get("CATMAP_AREA_MAX_RADIUS_M"), 10000)),
        area_max_sites=max(1, _to_int(get("CATMAP_AREA_MAX_SITES"), 50000)),
        maxsize_mb=_to_int(get("OVERPASS_MAXSIZE_MB"), 0),
        max_attempts=max(1, _to_int(get("OVERPASS_MAX_ATTEMPTS"), 2)),
        backoff_base_s=_to_float(get("OVERPASS_BACKOFF_S"), 0.5 if local else 5.0),
        backoff_max_s=_to_float(get("OVERPASS_BACKOFF_MAX_S"), 30.0 if local else 90.0),
        min_interval_s=_to_float(get("OVERPASS_MIN_INTERVAL_S"), 0.0 if local else 1.0),
        max_workers=max(1, _to_int(get("CATMAP_MAX_WORKERS"), 4 if local else 1)),
        cache_dir=Path(get("CATMAP_CACHE_DIR", str(ROOT / ".cache" / "overpass"))),
        cache_enabled=get("CATMAP_CACHE", "1").lower() not in ("0", "false", "no", "off"),
        cache_ttl_hours=ttl,
        cache_namespace=get("CATMAP_CACHE_NAMESPACE"),
        max_sites_per_company=_to_int(get("CATMAP_MAX_SITES_PER_COMPANY"), 50000),
        mf_api_key=get("METEOFRANCE_API_KEY"),
        mf_client_id=get("METEOFRANCE_CLIENT_ID"),
        mf_client_secret=get("METEOFRANCE_CLIENT_SECRET"),
        mf_token_url=get("METEOFRANCE_TOKEN_URL", "https://public-api.meteofrance.fr/token"),
        mf_base_url=get("METEOFRANCE_BASE_URL", "https://public-api.meteofrance.fr/public/DPVigilance/v1"),
        departements_geojson=get("CATMAP_DEPARTEMENTS_GEOJSON"),
    )
