"""Vigilance Météo-France (API DPVigilance) et croisement avec les sites géolocalisés.

Corrige A2 (secrets) et A17 (format) :
- aucun identifiant dans le code : clé API (``apikey``) ou OAuth2 client_credentials via
  l'environnement / ``st.secrets`` ;
- l'endpoint ``cartevigilance/encours`` renvoie un **JSON structuré** (``product.periods[]
  .timelaps.domain_ids[]`` avec ``max_color_id`` et ``phenomenon_items``) et non un GeoJSON ;
  ce format est analysé, avec repli sur un GeoJSON de type v1 ; en cas de format inconnu,
  l'erreur liste les clés reçues (diagnostic) ;
- les géométries de départements viennent d'un GeoJSON local/URL configurable, mis en cache ;
- plus de compteur incrémenté par effet de bord dans une fonction de style.

À valider sur votre compte : la structure ci-dessus est celle de la documentation de
l'API ; elle n'a pas pu être testée contre le service réel depuis l'environnement de
développement (pas d'accès réseau).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .polygon_index import PolygonIndex

log = logging.getLogger("catmap.vigilance")

COLORS = {1: "Vert", 2: "Jaune", 3: "Orange", 4: "Rouge"}
COLOR_HEX = {1: "#2e9e4f", 2: "#f2c200", 3: "#ef7d00", 4: "#d1202f"}
PHENOMENA = {
    1: "Vent violent", 2: "Pluie-inondation", 3: "Orages", 4: "Crues", 5: "Neige-verglas",
    6: "Canicule", 7: "Grand froid", 8: "Avalanches", 9: "Vagues-submersion",
}
DEFAULT_DEPARTEMENTS_URL = (
    "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
)


class VigilanceError(Exception):
    pass


class VigilanceFormatError(VigilanceError):
    pass


# --------------------------------------------------------------------------
# Authentification et appel
# --------------------------------------------------------------------------
class MeteoFranceAuth:
    """Clé API (prioritaire) ou OAuth2 client credentials avec renouvellement automatique."""

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None):
        self.s = settings
        self.session = session or requests.Session()
        self._token: Optional[str] = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.s.mf_api_key or (self.s.mf_client_id and self.s.mf_client_secret))

    def headers(self) -> dict:
        if self.s.mf_api_key:
            return {"apikey": self.s.mf_api_key, "Accept": "application/json"}
        if not (self.s.mf_client_id and self.s.mf_client_secret):
            raise VigilanceError(
                "Identifiants Météo-France absents : définir METEOFRANCE_API_KEY, ou "
                "METEOFRANCE_CLIENT_ID et METEOFRANCE_CLIENT_SECRET (variables d'environnement "
                "ou .streamlit/secrets.toml).")
        with self._lock:
            if self._token is None or time.monotonic() >= self._expires_at:
                try:
                    r = self.session.post(
                        self.s.mf_token_url, data={"grant_type": "client_credentials"},
                        auth=(self.s.mf_client_id, self.s.mf_client_secret), timeout=10)
                    r.raise_for_status()
                    tok = r.json()
                except (requests.RequestException, ValueError) as e:
                    raise VigilanceError(f"Obtention du jeton Météo-France impossible : {e}") from e
                if "access_token" not in tok:
                    raise VigilanceError("Réponse de jeton inattendue (access_token absent)")
                self._token = tok["access_token"]
                self._expires_at = time.monotonic() + max(60, int(tok.get("expires_in", 3600)) - 300)
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}


def fetch_vigilance(auth: MeteoFranceAuth, settings: Settings, session: Optional[requests.Session] = None) -> dict:
    sess = session or auth.session
    url = settings.mf_base_url.rstrip("/") + "/cartevigilance/encours"
    try:
        r = sess.get(url, headers=auth.headers(), timeout=15)
    except requests.RequestException as e:
        raise VigilanceError(f"Appel Météo-France impossible : {e}") from e
    if r.status_code in (401, 403):
        raise VigilanceError(f"Accès refusé (HTTP {r.status_code}) : vérifier l'abonnement à l'API DPVigilance et les identifiants.")
    if r.status_code != 200:
        raise VigilanceError(f"Erreur HTTP {r.status_code} : {r.text[:300]}")
    try:
        return r.json()
    except ValueError as e:
        raise VigilanceError("Réponse non JSON") from e


# --------------------------------------------------------------------------
# Outre-mer : chemins relatifs à la même API (mf_base_url), d'après le portail Météo-France
# --------------------------------------------------------------------------
OM_TERRITORIES = {
    "Polynésie française": {"path": "polynesie/cartevigilance/encours", "center": (-17.65, -149.45), "zoom": 7},
    "Nouvelle-Calédonie": {"path": "ncaledonie/cartevigilance/encours", "center": (-21.3, 165.5), "zoom": 6},
}
OM_DROM_PATH = "vigilanceom/flux/dernier"
OM_VIEWS = {
    "Guadeloupe": ((16.2, -61.55), 9), "Martinique": ((14.65, -61.0), 9), "Guyane": ((4.0, -53.0), 7),
    "La Réunion": ((-21.1, 55.5), 9), "Mayotte": ((-12.8, 45.15), 10), "Saint-Pierre-et-Miquelon": ((46.9, -56.3), 9),
    "Wallis-et-Futuna": ((-13.8, -177.8), 8),
}


def fetch_path(auth: MeteoFranceAuth, settings: Settings, path: str, session: Optional[requests.Session] = None,
               binary: bool = False):
    """GET d'un produit de l'API Vigilance (JSON par défaut, octets si ``binary``)."""
    sess = session or auth.session
    url = settings.mf_base_url.rstrip("/") + "/" + path.lstrip("/")
    try:
        r = sess.get(url, headers=auth.headers(), timeout=30 if binary else 15)
    except requests.RequestException as e:
        raise VigilanceError(f"Appel Météo-France impossible : {e}") from e
    if r.status_code in (401, 403):
        raise VigilanceError(f"Accès refusé (HTTP {r.status_code}) sur « {path} » : vérifier que l'abonnement couvre ce produit.")
    if r.status_code == 404:
        raise VigilanceError(f"Produit introuvable (HTTP 404) : « {path} ».")
    if r.status_code != 200:
        raise VigilanceError(f"Erreur HTTP {r.status_code} sur « {path} » : {r.text[:200]}")
    if binary:
        return r.content
    try:
        return r.json()
    except ValueError as e:
        raise VigilanceError("Réponse non JSON") from e


def inspect_zip(data: bytes, head_chars: int = 1200) -> list[dict]:
    """Liste le contenu d'un ZIP (nom, taille, début de texte) pour diagnostiquer un format non documenté."""
    import io
    import zipfile
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise VigilanceError("Le flux outre-mer n'est pas un ZIP valide") from e
    out = []
    for info in zf.infolist()[:50]:
        if info.is_dir():
            continue
        with zf.open(info) as fh:
            raw = fh.read(head_chars * 4)
        binary = raw.startswith(b"%PDF") or b"\x00" in raw or info.filename.lower().endswith((".pdf", ".png", ".jpg", ".zip"))
        out.append({"fichier": info.filename, "taille": info.file_size, "binaire": binary,
                    "debut": "" if binary else raw.decode("utf-8", "replace")[:head_chars]})
    return out


def parse_zip_json(data: bytes) -> list[pd.DataFrame]:
    """Si le ZIP contient des JSON au format « carte » reconnu, retourne leurs tables ; sinon liste vide."""
    import io
    import json
    import zipfile
    frames = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return frames
    for info in zf.infolist():
        if info.filename.lower().endswith(".json") and info.file_size < 20_000_000:
            try:
                frames.append(parse_vigilance(json.loads(zf.read(info)), phenomena={}))
            except (ValueError, VigilanceFormatError):
                continue
    return frames


# --------------------------------------------------------------------------
# Analyse de la réponse -> table longue
# --------------------------------------------------------------------------
COLUMNS = ["periode", "debut", "fin", "domaine", "phenomene_id", "phenomene", "couleur_id", "couleur"]


def _color_name(cid) -> str:
    try:
        return COLORS.get(int(cid), str(cid))
    except (TypeError, ValueError):
        return str(cid)


def parse_vigilance(payload: dict, phenomena: Optional[dict] = None) -> pd.DataFrame:
    """Table longue (une ligne par période × domaine × phénomène). Lève ``VigilanceFormatError``
    avec les clés rencontrées si le format n'est pas reconnu."""
    labels = PHENOMENA if phenomena is None else phenomena     # outre-mer : {} (les identifiants diffèrent de la métropole)
    rows: list[dict] = []
    periods = None
    if isinstance(payload, dict) and isinstance(payload.get("product"), dict):          # métropole
        periods = payload["product"].get("periods") or []
    elif isinstance(payload, dict) and isinstance(payload.get("timelaps"), dict):       # Polynésie, Nouvelle-Calédonie : une seule période
        periods = [{"echeance": "", "begin_validity_time": payload.get("update_time", ""),
                    "end_validity_time": payload.get("end_validity_time", ""), "timelaps": payload["timelaps"]}]
    if periods is not None:
        for per in periods:
            tl = per.get("timelaps") or {}
            for dom in tl.get("domain_ids") or []:
                base = {
                    "periode": per.get("echeance", ""), "debut": per.get("begin_validity_time", ""),
                    "fin": per.get("end_validity_time", ""), "domaine": str(dom.get("domain_id", "")).strip(),
                }
                items = dom.get("phenomenon_items") or []
                if not items:
                    rows.append({**base, "phenomene_id": None, "phenomene": "Tous phénomènes",
                                 "couleur_id": dom.get("max_color_id"), "couleur": _color_name(dom.get("max_color_id"))})
                for it in items:
                    pid = it.get("phenomenon_id")
                    cid = it.get("phenomenon_max_color_id")
                    try:
                        label = labels.get(int(pid), f"Phénomène {pid}")
                    except (TypeError, ValueError):
                        label = f"Phénomène {pid}"
                    rows.append({**base, "phenomene_id": pid, "phenomene": label,
                                 "couleur_id": cid, "couleur": _color_name(cid)})
    elif isinstance(payload, dict) and payload.get("type") == "FeatureCollection":  # repli : GeoJSON type v1
        for f in payload.get("features") or []:
            p = f.get("properties") or {}
            code = p.get("code") or p.get("NOM_DEP") or p.get("domain_id") or ""
            cid = p.get("color_id") or p.get("max_color_id")
            if cid is None and p.get("couleur"):
                cid = {v.lower(): k for k, v in COLORS.items()}.get(str(p["couleur"]).lower())
            rows.append({"periode": "", "debut": p.get("validite", ""), "fin": "", "domaine": str(code),
                         "phenomene_id": None, "phenomene": p.get("phenomene", "Tous phénomènes"),
                         "couleur_id": cid, "couleur": _color_name(cid)})
    else:
        keys = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        raise VigilanceFormatError(f"Format de vigilance non reconnu (clés de premier niveau : {keys})")
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["couleur_id"] = pd.to_numeric(df["couleur_id"], errors="coerce")
    return df


def worst_by_domain(df: pd.DataFrame, periode: Optional[str] = None) -> pd.DataFrame:
    """Couleur maximale et phénomènes en alerte (orange/rouge d'abord) par domaine."""
    d = df if periode is None else df[df["periode"] == periode]
    if d.empty:
        return pd.DataFrame(columns=["domaine", "couleur_id", "couleur", "phenomenes"])
    out = []
    for dom, g in d.groupby("domaine"):
        mx = g["couleur_id"].max()
        active = g[g["couleur_id"] >= 2].sort_values("couleur_id", ascending=False)
        phen = ", ".join(f"{r.phenomene} ({r.couleur})" for r in active.drop_duplicates("phenomene").itertuples())
        out.append({"domaine": dom, "couleur_id": mx, "couleur": _color_name(mx), "phenomenes": phen})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Départements (géométries) et exposition des sites
# --------------------------------------------------------------------------
def _domain_candidates(code: str) -> list[str]:
    c = str(code).strip().upper()
    out = [c]
    if c.isdigit():
        out.append(c.lstrip("0") or "0")
    if c in ("2A", "2B"):
        out.append("20")
    return out


class DepartementIndex:
    def __init__(self, geojson: dict):
        self.geojson = geojson
        feats = geojson.get("features") or []
        rings, self.codes, self.noms = [], [], []
        for f in feats:
            g = f.get("geometry") or {}
            p = f.get("properties") or {}
            coords = g.get("coordinates") or []
            polys = coords if g.get("type") == "MultiPolygon" else [coords]
            shape = [np.asarray(r, dtype=float)[:, :2] for poly in polys for r in poly]
            rings.append(shape)
            self.codes.append(str(p.get("code") or p.get("INSEE_DEP") or p.get("code_insee") or "").strip().upper())
            self.noms.append(str(p.get("nom") or p.get("NOM_DEP") or p.get("name") or ""))
        if not rings:
            raise VigilanceError("GeoJSON des départements vide ou invalide")
        self._index = PolygonIndex(rings)

    def locate(self, lon, lat, snap_km: float = 5.0) -> pd.DataFrame:
        idx, _, _ = self._index.locate(lon, lat, snap_km=snap_km)
        found = idx >= 0
        safe = np.where(found, idx, 0)
        return pd.DataFrame({
            "departement_code": np.where(found, np.array(self.codes, dtype=object)[safe], None),
            "departement": np.where(found, np.array(self.noms, dtype=object)[safe], None),
        })


def load_departements(settings: Settings, cache_dir: Optional[Path] = None,
                      session: Optional[requests.Session] = None) -> DepartementIndex:
    """Charge le GeoJSON des départements (chemin local ou URL), avec cache disque."""
    src = settings.departements_geojson or DEFAULT_DEPARTEMENTS_URL
    if not src.startswith(("http://", "https://")):
        return DepartementIndex(json.loads(Path(src).read_text(encoding="utf-8")))
    cache = (cache_dir or Path(settings.cache_dir).parent) / "departements.geojson"
    if cache.exists():
        try:
            return DepartementIndex(json.loads(cache.read_text(encoding="utf-8")))
        except (ValueError, VigilanceError):
            cache.unlink(missing_ok=True)
    try:
        r = (session or requests).get(src, timeout=30)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        raise VigilanceError(f"Téléchargement des départements impossible ({src}) : {e}") from e
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return DepartementIndex(data)


def expose_sites(sites: pd.DataFrame, vigilance: pd.DataFrame, deps: DepartementIndex,
                 periode: Optional[str] = None) -> pd.DataFrame:
    """Ajoute à chaque site situé en France métropolitaine le département et la vigilance
    en cours (couleur max et phénomènes). Les autres sites gardent des valeurs vides."""
    out = sites.copy()
    for c in ("departement_code", "departement", "vigilance_couleur_id", "vigilance_couleur", "vigilance_phenomenes"):
        out[c] = None
    if out.empty:
        return out
    mask = out["pays_iso2"].eq("FR") if "pays_iso2" in out else pd.Series(True, index=out.index)
    if not mask.any():
        return out
    sub = out[mask]
    loc = deps.locate(sub["long"].to_numpy(float), sub["lat"].to_numpy(float))
    worst = worst_by_domain(vigilance, periode)
    lookup = {str(r.domaine).strip().upper(): r for r in worst.itertuples()}
    cols = {"couleur_id": [], "couleur": [], "phenomenes": []}
    for code in loc["departement_code"]:
        hit = None
        if code:
            for cand in _domain_candidates(code):
                if cand in lookup:
                    hit = lookup[cand]
                    break
        cols["couleur_id"].append(None if hit is None else hit.couleur_id)
        cols["couleur"].append(None if hit is None else hit.couleur)
        cols["phenomenes"].append(None if hit is None else (hit.phenomenes or None))
    out.loc[mask, "departement_code"] = loc["departement_code"].to_numpy()
    out.loc[mask, "departement"] = loc["departement"].to_numpy()
    out.loc[mask, "vigilance_couleur_id"] = cols["couleur_id"]
    out.loc[mask, "vigilance_couleur"] = cols["couleur"]
    out.loc[mask, "vigilance_phenomenes"] = cols["phenomenes"]
    return out
