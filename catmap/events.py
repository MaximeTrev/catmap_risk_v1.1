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


# =========================================================================== v2 : détails des couches, clic, légende, dates, croisement
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlencode

import numpy as np

# Description « humaine » déduite du nom des couches (la description officielle du service est affichée à côté).
LAYER_NOTES = (
    ("modis.ba.poly", "Zones brûlées détectées par satellite (MODIS), dessinées en polygones. Même information que « Zones brûlées (EFFIS) » "
                      "mais servie sous forme d'image par le service cartographique (non cliquable ici)."),
    ("viirs.ba", "Zones brûlées détectées par satellite (VIIRS), sous forme d'image."),
    ("modis.ba", "Zones brûlées détectées par satellite (MODIS), sous forme d'image : chaque carré coloré est une maille de détection, "
                 "la couleur indique la date ou l'ancienneté de la détection (voir la légende)."),
    ("hotspot", "Foyers actifs : anomalies thermiques détectées par satellite (points chauds), les plus récentes."),
    (".hs", "Foyers actifs : anomalies thermiques détectées par satellite (points chauds)."),
    ("fwi", "Indice Forêt-Météo (Fire Weather Index) : danger météorologique d'incendie prévu, de « très faible » à « extrême ». "
            "Ce n'est pas un incendie constaté mais un risque : plus la valeur est haute, plus un départ de feu se propagerait vite."),
    ("ffmc", "Composante de l'indice Forêt-Météo : humidité des litières fines (facilité d'allumage)."),
    ("dmc", "Composante de l'indice Forêt-Météo : humidité de la couche d'humus (feux profonds)."),
    ("dc", "Composante de l'indice Forêt-Météo : sécheresse des couches profondes (feux de longue durée)."),
    ("isi", "Composante de l'indice Forêt-Météo : vitesse de propagation initiale."),
    ("bui", "Composante de l'indice Forêt-Météo : quantité de combustible disponible."),
)

# Seuils usuels de l'indice Forêt-Météo EFFIS (valeur supérieure de chaque classe).
FWI_CLASSES = ((5.2, "Très faible"), (11.2, "Faible"), (21.3, "Modéré"), (38.0, "Élevé"), (50.0, "Très élevé"), (math.inf, "Extrême"))


def layer_note(name: str, title: str = "") -> str:
    low = f"{name} {title}".lower()
    for key, note in LAYER_NOTES:
        if key in low:
            return note
    return ""


def fwi_class(value) -> tuple[int, str]:
    """(rang 1..6, libellé) d'une valeur d'indice Forêt-Météo ; (0, '') si la valeur n'est pas exploitable."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0, ""
    if not math.isfinite(v) or v < 0:
        return 0, ""
    for i, (hi, label) in enumerate(FWI_CLASSES, 1):
        if v < hi:
            return i, label
    return len(FWI_CLASSES), FWI_CLASSES[-1][1]


def fetch_layer_details(base: str, session=None) -> list[dict]:
    """Couches nommées avec description, dimension temporelle et caractère « interrogeable » : [{name, title, abstract, time, queryable}]."""
    try:
        r = _get(session, base, {"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"})
        root = ET.fromstring(r.text.encode("utf-8"))
    except (requests.RequestException, ET.ParseError, AttributeError):
        return []
    out, seen = [], set()
    for el in root.iter():
        if el.tag.split("}")[-1] != "Layer":
            continue
        d = {"name": None, "title": "", "abstract": "", "time": "", "queryable": str(el.get("queryable", "")) == "1"}
        for ch in el:
            t = ch.tag.split("}")[-1]
            txt = (ch.text or "").strip()
            if t == "Name" and txt:
                d["name"] = txt
            elif t == "Title":
                d["title"] = txt
            elif t == "Abstract":
                d["abstract"] = txt
            elif t in ("Dimension", "Extent") and (ch.get("name") or "").lower() == "time":
                d["time"] = txt or str(ch.get("default", ""))
        if d["name"] and d["name"] not in seen:
            seen.add(d["name"])
            out.append(d)
    return out


def legend_url(base: str, layer: str) -> str:
    sep = "&" if "?" in base else "?"
    return base + sep + urlencode({"service": "WMS", "request": "GetLegendGraphic", "version": "1.1.1",
                                   "format": "image/png", "layer": layer})


_NUM = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def first_number(text: str, prefer: tuple[str, ...] = ("value", "fwi", "pixel", "gray", "band")) -> Optional[float]:
    """Première valeur numérique d'une réponse GetFeatureInfo : lignes « clé = valeur » ou « "clé": valeur » (celles qui
    évoquent une valeur d'abord), sinon un nombre seul."""
    if not text:
        return None
    body = text.strip()
    if _NUM.fullmatch(body):
        return float(body.replace(",", "."))
    pairs = [(m.group(1).lower(), m.group(2)) for m in
             re.finditer(r"([A-Za-z_][\w .-]*?)[\"']?\s*[:=]\s*[\"']?(-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)", body)]
    pairs.sort(key=lambda kv: 0 if any(p in kv[0] for p in prefer) else 1)
    for _, num in pairs:
        try:
            return float(num.replace(",", "."))
        except ValueError:
            continue
    return None


def _fi_bbox(lat: float, lon: float, zoom: float, size: int = 101):
    deg_px = 360.0 / (256.0 * (2 ** float(zoom)))
    half = (size // 2 + 0.5) * deg_px
    half_lat = half * max(math.cos(math.radians(lat)), 0.05)
    return lat - half_lat, lon - half, lat + half_lat, lon + half


def get_feature_info(base: str, layer: str, lat: float, lon: float, zoom: float = 6, session=None,
                     time_value: Optional[str] = None) -> tuple[str, list[dict]]:
    """Valeur d'une couche WMS au point (lat, lon) : (texte brut de la réponse, diagnostic). Texte vide si rien."""
    diag = []
    south, west, north, east = _fi_bbox(lat, lon, zoom)
    for fmt in ("text/plain", "application/json", "text/html", "application/vnd.ogc.gml"):
        params = {"service": "WMS", "version": "1.3.0", "request": "GetFeatureInfo", "layers": layer, "query_layers": layer,
                  "styles": "", "crs": "EPSG:4326", "bbox": f"{south},{west},{north},{east}", "width": 101, "height": 101,
                  "i": 50, "j": 50, "info_format": fmt, "feature_count": 5}
        if time_value:
            params["time"] = time_value
        try:
            r = _get(session, base, params)
        except requests.RequestException as e:
            diag.append(_diag(f"{base} [{layer}] {fmt}", "erreur", str(e)[:150]))
            return "", diag
        body = (getattr(r, "text", "") or "").strip()
        if r.status_code != 200:
            diag.append(_diag(f"{base} [{layer}] {fmt}", r.status_code, "refusé"))
            continue
        if "ServiceException" in body[:500]:
            diag.append(_diag(f"{base} [{layer}] {fmt}", 200, "format ou couche non interrogeable"))
            continue
        diag.append(_diag(f"{base} [{layer}] {fmt}", 200, "ok" if body else "réponse vide"))
        if body:
            return body, diag
    return "", diag


def sample_layer(base: str, layer: str, lats, lons, session=None, time_value: Optional[str] = None, step: float = 0.05,
                 max_cells: int = 300, workers: int = 8) -> tuple[np.ndarray, dict]:
    """Valeur numérique de la couche pour chaque point (NaN si inconnue). Les points sont regroupés par maille de `step` degrés
    pour limiter les appels ; au plus `max_cells` mailles sont interrogées. Retourne (valeurs, bilan)."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    ok = np.isfinite(lats) & np.isfinite(lons)
    keys = np.where(ok, np.round(lats / step).astype(float) * step, np.nan), np.where(ok, np.round(lons / step).astype(float) * step, np.nan)
    cells = list(dict.fromkeys(zip(keys[0][ok].round(4), keys[1][ok].round(4))))
    truncated = len(cells) > max_cells
    cells = cells[:max_cells]

    def one(c):
        txt, _ = get_feature_info(base, layer, float(c[0]), float(c[1]), zoom=10, session=session, time_value=time_value)
        return c, first_number(txt)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = dict(ex.map(one, cells))
    vals = np.full(len(lats), np.nan)
    for i in np.where(ok)[0]:
        vals[i] = got.get((round(float(keys[0][i]), 4), round(float(keys[1][i]), 4)), np.nan) or np.nan
    n_ok = sum(1 for v in got.values() if v is not None)
    return vals, {"mailles": len(cells), "valeurs_lues": n_ok, "tronque": truncated}


# --------------------------------------------------------------------------- zones brûlées : dates et filtre
def detect_date_columns(table: pd.DataFrame) -> list[str]:
    """Colonnes dont le nom évoque une date ou dont le contenu s'interprète majoritairement comme des dates récentes."""
    cols = []
    for c in table.columns:
        if str(c).lower() == "geometry":
            continue
        name_hint = any(w in str(c).lower() for w in ("date", "time", "day", "initial", "final", "update"))
        sample = table[c].dropna().head(50)
        if sample.empty:
            continue
        if pd.api.types.is_numeric_dtype(sample) and not name_hint:
            continue
        parsed = _to_dt(sample)
        if parsed.notna().mean() >= 0.8 and (name_hint or not pd.api.types.is_numeric_dtype(sample)):
            cols.append(c)
    return cols


def _to_dt(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        s = series.astype("float64")
        unit = "ms" if s.dropna().abs().median() > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def default_date_columns(cols: list[str]) -> tuple[Optional[str], Optional[str]]:
    """(colonne de début, colonne de fin) probables parmi les colonnes de date."""
    low = {c: str(c).lower() for c in cols}
    start = next((c for c in cols if any(w in low[c] for w in ("initial", "start", "first", "firedate"))), None)
    end = next((c for c in cols if any(w in low[c] for w in ("final", "last", "end", "update"))), None)
    if start is None and cols:
        start = cols[0]
    if end == start:
        end = None
    return start, end


def _utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def filter_burnt(gj: Optional[dict], mode: str, start_col: Optional[str], end_col: Optional[str],
                 d1=None, d2=None, active_days: int = 3, now: Optional[pd.Timestamp] = None) -> Optional[dict]:
    """Garde les zones brûlées selon le mode : « tout », « en cours » (dernière date >= maintenant - active_days jours),
    « date » (zone active ce jour-là) ou « période » (activité chevauchant [d1, d2])."""
    if not gj or mode == "tout" or not start_col:
        return gj
    feats = gj.get("features") or []
    if not feats:
        return gj
    props = pd.DataFrame([(f.get("properties") or {}) for f in feats])
    if start_col not in props.columns:
        return gj
    a = _to_dt(props[start_col])
    b = _to_dt(props[end_col]) if end_col and end_col in props.columns else a
    b = b.fillna(a)
    now = now or pd.Timestamp.now(tz="UTC")
    if mode == "en cours":
        keep = b >= now - pd.Timedelta(days=active_days)
    else:
        lo = _utc(d1)
        hi = _utc(d2 if (mode == "période" and d2 is not None) else d1) + pd.Timedelta(days=1)
        keep = (a < hi) & (b >= lo)
    keep = keep.fillna(False).to_numpy()
    return {**gj, "features": [f for f, k in zip(feats, keep) if k]}


def rings_from_feature(feature: dict) -> list[np.ndarray]:
    """Anneaux (lon, lat) d'une entité Polygon / MultiPolygon."""
    g = (feature or {}).get("geometry") or {}
    t, c = g.get("type"), g.get("coordinates")
    rings: list[np.ndarray] = []
    try:
        if t == "Polygon":
            rings = [np.asarray(r, dtype=float)[:, :2] for r in c]
        elif t == "MultiPolygon":
            rings = [np.asarray(r, dtype=float)[:, :2] for poly in c for r in poly]
    except (ValueError, IndexError, TypeError):
        return []
    return rings


def cross_with_burnt(df: pd.DataFrame, lat_c: str, lon_c: str, gj: Optional[dict], snap_km: float = 0.0,
                     date_cols: tuple = ()) -> pd.DataFrame:
    """Ajoute à df : incendie_statut (« Dans une zone brûlée » / « À moins de X km » / « Non touché »), incendie_distance_km,
    et les dates / surface de la zone brûlée associée."""
    from .polygon_index import PolygonIndex
    out = df.copy()
    lat = pd.to_numeric(out[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()
    lon = pd.to_numeric(out[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()
    feats = (gj or {}).get("features") or []
    out["incendie_statut"] = "Non touché"
    out["incendie_distance_km"] = np.nan
    if not feats:
        return out
    idx = PolygonIndex([rings_from_feature(f) for f in feats])
    ids, methods, dist = idx.locate(lon, lat, snap_km=snap_km)
    props = [f.get("properties") or {} for f in feats]
    extra = [c for c in date_cols if c] + [k for k in props[0] if "area" in str(k).lower() or "ha" == str(k).lower()][:1]
    for c in dict.fromkeys(extra):
        out[f"incendie_{c}"] = [props[i].get(c) if i >= 0 else None for i in ids]
    inside = np.array([m == "within" for m in methods])
    near = np.array([m == "nearest" for m in methods])
    out.loc[inside, "incendie_statut"] = "Dans une zone brûlée"
    out.loc[near, "incendie_statut"] = [f"À moins de {snap_km:g} km" for _ in range(int(near.sum()))]
    out["incendie_distance_km"] = np.where(inside | near, np.round(np.nan_to_num(dist, nan=0.0), 2), np.nan)
    return out
