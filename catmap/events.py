"""Suivi d'évènements climatiques : inondations, incendies, activations Copernicus EMS.

Trois sources publiques, sans clé :

* Copernicus EMS Rapid Mapping : liste des activations (API JSON paginée)
  https://mapping.emergency.copernicus.eu/activations/api/activations/
* EFFIS (incendies) : couches WMS (cartes) et WFS (périmètres brûlés « nrt.ba.poly »).
* EFAS (Europe) et GloFAS (monde) : couches WMS de crues.

Les adresses et noms de couches ci-dessous proviennent de la documentation publique mais n'ont pas pu être appelés
depuis l'environnement de développement : chaque appel renvoie donc un diagnostic (adresse essayée, statut, erreur) et
les couches WMS sont découvertes dynamiquement via GetCapabilities plutôt que codées en dur.

Le module ne dépend pas de Streamlit ; tout est testable avec une fausse session HTTP (méthode get).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

import pandas as pd
import requests

EMS_API = "https://mapping.emergency.copernicus.eu/activations/api/activations/"
EMS_PAGE = "https://mapping.emergency.copernicus.eu/activations/{code}/"

# Plusieurs hôtes sont documentés : on essaie dans l'ordre et on garde le premier qui répond.
EFFIS_WMS = ("https://maps.effis.emergency.copernicus.eu/effis", "https://ies-ows.jrc.ec.europa.eu/effis")
EFFIS_WFS = ("https://maps.effis.emergency.copernicus.eu/gwis", "https://ies-ows.jrc.ec.europa.eu/gwis")
EFAS_WMS = ("https://european-flood.emergency.copernicus.eu/api/wms/", "https://www.efas.eu/api/wms/")
GLOFAS_WMS = ("https://ows.globalfloods.eu/glofas-ows/ows.py",)
BURNT_AREA_LAYER = "nrt.ba.poly"

TIMEOUT = 30
UA = {"User-Agent": "CatMap-risk/1.0"}

BBOX_PRESETS = {                      # (ouest, sud, est, nord)
    "Europe": (-12.0, 34.0, 32.0, 62.0),
    "France métropolitaine": (-5.5, 41.0, 10.0, 51.5),
    "Bassin méditerranéen": (-10.0, 30.0, 36.0, 47.0),
    "Péninsule ibérique": (-10.5, 35.5, 4.5, 44.5),
    "Grèce et Balkans": (13.0, 34.0, 30.0, 46.0),
}

FLOOD_WORDS = ("flood", "inond", "storm surge")
FIRE_WORDS = ("fire", "wildfire", "incend", "forest")
KIND_LABEL = {"flood": "Inondation", "fire": "Incendie", "other": "Autre"}


def _get(session, url: str, params: Optional[dict] = None):
    s = session or requests
    return s.get(url, params=params, timeout=TIMEOUT, headers=UA)


def _diag(url: str, status, note: str = "") -> dict:
    return {"url": url, "statut": status, "remarque": note}


# --------------------------------------------------------------------------- Copernicus EMS Rapid Mapping
_POINT = re.compile(r"POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)", re.I)


def parse_point(wkt) -> Optional[tuple[float, float]]:
    """« SRID=4326;POINT (lon lat) » ou « POINT (lon lat) » -> (lat, lon)."""
    if not isinstance(wkt, str):
        return None
    m = _POINT.search(wkt)
    if not m:
        return None
    lon, lat = float(m.group(1)), float(m.group(2))
    if abs(lat) > 90 or abs(lon) > 180:
        return None
    return lat, lon


def hazard_kind(category: str, name: str = "") -> str:
    text = f"{category} {name}".lower()
    if any(w in text for w in FLOOD_WORDS):
        return "flood"
    if any(w in text for w in FIRE_WORDS):
        return "fire"
    return "other"


def _cat(item: dict) -> str:
    c = item.get("category")
    if isinstance(c, dict):
        return str(c.get("name") or c.get("slug") or "")
    return str(c or "")


def parse_activations(results: list[dict]) -> pd.DataFrame:
    rows = []
    for it in results or []:
        if not isinstance(it, dict) or not it.get("code"):
            continue
        pt = parse_point(it.get("centroid"))
        countries = ", ".join(str(c.get("short_name") or c.get("name") or "") for c in (it.get("countries") or [])
                              if isinstance(c, dict)).strip(", ")
        cat = _cat(it)
        rows.append({
            "code": it["code"], "nom": it.get("name") or "", "type": cat,
            "genre": hazard_kind(cat, it.get("name") or ""),
            "pays": countries, "lat": pt[0] if pt else None, "long": pt[1] if pt else None,
            "activation": pd.to_datetime(it.get("activationTime"), utc=True, errors="coerce"),
            "maj": pd.to_datetime(it.get("lastUpdate"), utc=True, errors="coerce"),
            "cloture": bool(it.get("closed")), "phase": it.get("drmPhase") or "",
            "cartes": it.get("n_products"), "lien": EMS_PAGE.format(code=it["code"])})
    df = pd.DataFrame(rows, columns=["code", "nom", "type", "genre", "pays", "lat", "long", "activation", "maj", "cloture",
                                     "phase", "cartes", "lien"])
    return df.sort_values("activation", ascending=False, na_position="last").reset_index(drop=True)


def fetch_activations(session=None, *, max_pages: int = 6, limit: int = 100) -> tuple[pd.DataFrame, list[dict]]:
    """Dernières activations Rapid Mapping. Retourne (tableau, diagnostic). N'échoue jamais : tableau vide + diagnostic."""
    diag, results = [], []
    url, params = EMS_API, {"limit": limit, "offset": 0, "ordering": "-activationTime"}
    total = None
    for _ in range(max_pages):
        try:
            r = _get(session, url, params)
        except requests.RequestException as e:
            diag.append(_diag(url, "erreur", str(e)[:200]))
            break
        if r.status_code != 200:
            diag.append(_diag(url, r.status_code, "réponse inattendue"))
            break
        try:
            data = r.json()
        except ValueError:
            diag.append(_diag(url, 200, "réponse non JSON"))
            break
        page = data.get("results") if isinstance(data, dict) else data
        if not isinstance(page, list):
            diag.append(_diag(url, 200, "format inattendu (pas de « results »)"))
            break
        results += page
        total = data.get("count") if isinstance(data, dict) else None
        diag.append(_diag(url, 200, f"{len(page)} activations"))
        nxt = data.get("next") if isinstance(data, dict) else None
        if not nxt or not page:
            break
        url, params = nxt, None                      # l'URL « next » contient déjà limit/offset
    df = parse_activations(results)
    if total and total > len(results):
        diag.append(_diag(EMS_API, "info", f"{len(results)} activations lues sur {total} (les plus récentes d'abord si le tri est pris en compte)"))
    return df, diag


def filter_activations(df: pd.DataFrame, kinds: list[str], days: Optional[int], only_open: bool = False,
                       now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    out = df[df["genre"].isin(kinds)] if kinds else df
    if days:
        now = now or pd.Timestamp.now(tz="UTC")
        out = out[out["activation"] >= now - pd.Timedelta(days=days)]
    if only_open:
        out = out[~out["cloture"]]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- WMS : découverte des couches
def parse_capabilities(xml_text: str) -> list[tuple[str, str]]:
    """Couches nommées d'un document GetCapabilities (WMS 1.1/1.3) : [(nom, titre)]."""
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        return []
    out, seen = [], set()
    for el in root.iter():
        if el.tag.split("}")[-1] != "Layer":
            continue
        name = title = None
        for ch in el:
            t = ch.tag.split("}")[-1]
            if t == "Name" and ch.text:
                name = ch.text.strip()
            elif t == "Title" and ch.text:
                title = ch.text.strip()
        if name and name not in seen:
            seen.add(name)
            out.append((name, title or name))
    return out


def discover_wms(urls, session=None) -> tuple[Optional[str], list[tuple[str, str]], list[dict]]:
    """Premier hôte qui répond à GetCapabilities : (url, couches, diagnostic)."""
    diag = []
    for base in urls:
        params = {"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"}
        try:
            r = _get(session, base, params)
        except requests.RequestException as e:
            diag.append(_diag(base, "erreur", str(e)[:200]))
            continue
        if r.status_code != 200:
            diag.append(_diag(base, r.status_code, "GetCapabilities refusé"))
            continue
        layers = parse_capabilities(r.text)
        if not layers:
            diag.append(_diag(base, 200, "aucune couche lisible dans la réponse"))
            continue
        diag.append(_diag(base, 200, f"{len(layers)} couches"))
        return base, layers, diag
    return None, [], diag


def pick_layers(layers: list[tuple[str, str]], words: tuple[str, ...]) -> list[str]:
    """Noms de couches dont le nom ou le titre contient un des mots (insensible à la casse)."""
    return [n for n, t in layers if any(w in f"{n} {t}".lower() for w in words)]


FIRE_LAYER_WORDS = ("hotspot", "active fire", ".hs", "burnt", "burned", "fwi", "fire danger", "ba.")
FLOOD_LAYER_WORDS = ("flood", "alert", "rep", "discharge", "probab", "thresh", "hazard")


# --------------------------------------------------------------------------- EFFIS : périmètres brûlés (WFS)
def burnt_areas(bbox: tuple[float, float, float, float], session=None, urls=EFFIS_WFS, max_features: int = 2000
                ) -> tuple[Optional[dict], list[dict]]:
    """Périmètres de zones brûlées récents (GeoJSON) dans la zone ; None si aucun hôte n'a répondu."""
    diag = []
    w, s, e, n = bbox
    for base in urls:
        for fmt in ("geojson", "application/json"):
            params = {"service": "WFS", "version": "1.1.0", "request": "GetFeature", "typename": BURNT_AREA_LAYER,
                      "outputformat": fmt, "maxfeatures": max_features, "bbox": f"{w},{s},{e},{n},EPSG:4326"}
            try:
                r = _get(session, base, params)
            except requests.RequestException as ex:
                diag.append(_diag(base, "erreur", str(ex)[:200]))
                break                                  # hôte injoignable : inutile d'essayer l'autre format
            if r.status_code != 200:
                diag.append(_diag(f"{base} ({fmt})", r.status_code, "requête refusée"))
                continue
            try:
                data = r.json()
            except ValueError:
                diag.append(_diag(f"{base} ({fmt})", 200, "réponse non JSON (format ou couche non reconnus)"))
                continue
            feats = data.get("features") if isinstance(data, dict) else None
            if not isinstance(feats, list):
                diag.append(_diag(f"{base} ({fmt})", 200, "pas de « features » dans la réponse"))
                continue
            note = f"{len(feats)} périmètres"
            if len(feats) >= max_features:
                note += " (limite atteinte : zone tronquée, réduisez la zone)"
            diag.append(_diag(f"{base} ({fmt})", 200, note))
            return data, diag
    return None, diag


def burnt_table(geojson: Optional[dict]) -> pd.DataFrame:
    """Attributs des périmètres sous forme de tableau (colonnes telles que fournies par le service)."""
    feats = (geojson or {}).get("features") or []
    rows = [dict((f.get("properties") or {})) for f in feats if isinstance(f, dict)]
    return pd.DataFrame(rows)
