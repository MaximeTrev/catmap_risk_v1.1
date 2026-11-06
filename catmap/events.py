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

VERSION = 4   # incrémenté à chaque évolution (v3 : contours des activations ; v4 : croisements par couche, légendes de secours)

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
    ("floodprotection", "Niveau de protection contre les crues : période de retour (en années) de la crue que les ouvrages existants "
                        "sont censés supporter ; 200 = protégé contre une crue qui revient en moyenne tous les 200 ans. "
                        "Ce n'est pas une zone inondée."),
    ("flood_scenarios", "Zones inondables des scénarios de crue : un point situé dans une zone peut être inondé pour un des scénarios "
                        "de crue modélisés (voir la légende pour leur signification)."),
    ("floodhazard", "Aléa d'inondation : zones susceptibles d'être inondées pour une crue de période de retour donnée (ex. « 100y » = crue "
                    "centennale). Une valeur au-dessus de 0 signifie que le point est dans la zone d'aléa (voir la légende pour la signification exacte)."),
    ("accrain", "Précipitations cumulées prévues (en mm) sur la période indiquée : un fort cumul peut provoquer des crues."),
    ("reppoints", "Points de référence (stations) où le débit prévu est comparé aux seuils de crue (périodes de retour 2, 5, 20 ans) ; "
                  "la couleur donne le niveau d'alerte."),
    ("countries", "Frontières des pays (fond de repère)."),
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
        d = {"name": None, "title": "", "abstract": "", "time": "", "time_default": "", "queryable": str(el.get("queryable", "")) == "1", "bbox": None, "legends": [], "styles": []}
        for ch in el:
            t = ch.tag.split("}")[-1]
            txt = (ch.text or "").strip()
            if t == "Style":
                for sub in ch.iter():
                    st_ = sub.tag.split("}")[-1]
                    if st_ == "Name" and (sub.text or "").strip():
                        d["styles"].append(sub.text.strip())
                    elif st_ == "OnlineResource":
                        href = next((v for k, v in sub.attrib.items() if k.endswith("href")), "")
                        if href:
                            d["legends"].append(href.replace("&amp;", "&"))
            if t == "EX_GeographicBoundingBox":
                v = {c.tag.split("}")[-1]: (c.text or "").strip() for c in ch}
                try:
                    d["bbox"] = (float(v["westBoundLongitude"]), float(v["southBoundLatitude"]),
                                 float(v["eastBoundLongitude"]), float(v["northBoundLatitude"]))
                except (KeyError, ValueError):
                    pass
            elif t == "LatLonBoundingBox" and d["bbox"] is None:
                try:
                    d["bbox"] = (float(ch.get("minx")), float(ch.get("miny")), float(ch.get("maxx")), float(ch.get("maxy")))
                except (TypeError, ValueError):
                    pass
            if t == "Name" and txt:
                d["name"] = txt
            elif t == "Title":
                d["title"] = txt
            elif t == "Abstract":
                d["abstract"] = txt
            elif t in ("Dimension", "Extent") and (ch.get("name") or "").lower() == "time":
                d["time"] = txt or str(ch.get("default", ""))
                d["time_default"] = str(ch.get("default", "") or "")
        if d["name"] and d["name"] not in seen:
            seen.add(d["name"])
            out.append(d)
    return out


def legend_url(base: str, layer: str) -> str:
    sep = "&" if "?" in base else "?"
    return base + sep + urlencode({"service": "WMS", "request": "GetLegendGraphic", "version": "1.1.1",
                                   "format": "image/png", "layer": layer, "layers": layer})


def getmap_url(base: str, layer: str, bbox=(-12.0, 34.0, 32.0, 62.0), time_value: Optional[str] = None) -> str:
    """Adresse d'une image GetMap de la couche : à ouvrir dans le navigateur pour voir l'éventuelle erreur du service
    (couche inconnue, jeton exigé, date hors plage…)."""
    w, s_, e, n = bbox
    params = {"service": "WMS", "version": "1.3.0", "request": "GetMap", "layers": layer, "styles": "", "crs": "EPSG:4326",
              "bbox": f"{s_},{w},{n},{e}", "width": 800, "height": 500, "format": "image/png", "transparent": "true"}
    if time_value:
        params["time"] = time_value
    return base + ("&" if "?" in base else "?") + urlencode(params)


def fetch_legend(base: str, layer: str, legends=(), styles=(), session=None) -> tuple[Optional[bytes], str, list[dict]]:
    """Image de légende d'une couche : (octets, type MIME, diagnostic). Essaie les adresses publiées dans le catalogue du service,
    puis plusieurs variantes de GetLegendGraphic ; (None, '', diag) si aucune ne renvoie une image."""
    cands = list(legends)
    bases = [base] + ([base.rstrip("/")] if base.endswith("/") else [])
    for b in bases:
        sep = "&" if "?" in b else "?"
        for ver in ("1.1.0", "1.0.0", "1.3.0"):
            for sty in (list(styles)[:1] or [None]):
                for extra in ({}, {"sld_version": "1.1.0"}, {"layers": layer, "sld_version": "1.1.0"}):
                    p = {"request": "GetLegendGraphic", "version": ver, "format": "image/png", "layer": layer, **extra}
                    if ver != "1.0.0":
                        p["service"] = "WMS"
                    if sty:
                        p["style"] = sty
                    cands.append(b + sep + urlencode(p))
    diag = []
    for u in dict.fromkeys(cands):
        try:
            r = (session or requests).get(u, timeout=TIMEOUT, headers=UA)
        except requests.RequestException as e:
            diag.append(_diag(u, "erreur", str(e)[:150]))
            continue
        ctype = (getattr(r, "headers", {}) or {}).get("Content-Type", "") or ""
        content = getattr(r, "content", None)
        if r.status_code == 200 and content and ctype.lower().startswith("image"):
            diag.append(_diag(u, 200, "image"))
            return content, ctype, diag
        diag.append(_diag(u, r.status_code, (getattr(r, "text", "") or "")[:120].replace("\n", " ") or f"type {ctype or '?'}"))
    return None, "", diag


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
    pairs += [(m.group(1).lower(), m.group(2)) for m in
              re.finditer(r"<([A-Za-z_][\w:.-]*)[^>]*>\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*<", body)]
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


FI_FORMATS = ("text/plain", "application/json", "text/html", "application/vnd.ogc.gml", "text/xml", "application/geojson")


def get_feature_info(base: str, layer: str, lat: float, lon: float, zoom: float = 6, session=None,
                     time_value: Optional[str] = None, formats=None, exhaustive: bool = False) -> tuple[str, list[dict]]:
    """Valeur d'une couche WMS au point (lat, lon) : (texte brut de la réponse, diagnostic). Texte vide si rien.
    Par défaut une réponse valide mais vide est définitive ; avec `exhaustive`, on essaie les autres formats tant que la réponse
    est vide (utile pour un clic isolé). `formats` impose la liste des formats essayés."""
    diag = []
    south, west, north, east = _fi_bbox(lat, lon, zoom)
    for fmt in (formats or FI_FORMATS):
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
        diag.append(_diag(f"{base} [{layer}] {fmt}", 200, "ok" if body else "réponse vide (rien à cet endroit)"))
        if body or not exhaustive:
            return body, diag                 # réponse valide, même vide : inutile d'essayer les autres formats
    return "", diag


_NO_FEATURE = ("no features", "no feature", "returned no results", "no results", "nothing found", "pas de résultat")


def _json_nonempty(x) -> bool:
    if isinstance(x, list):
        return any(_json_nonempty(i) for i in x)
    if isinstance(x, dict):
        if x.get("type") == "layer_tab":
            return False
        if x.get("type") == "table":
            return bool(((x.get("data") or {}).get("rows")))
        if isinstance(x.get("features"), list):
            return bool(x["features"])
        return bool(x)
    return x not in (None, "", 0)


def info_has_feature(text: str) -> bool:
    """La réponse GetFeatureInfo décrit-elle au moins un objet à cet endroit (même sans valeur numérique) ?
    « [] », une collection vide ou une page vide signifient : rien ici."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if any(w in low for w in _NO_FEATURE):
        return False
    if low[0] in "{[":
        try:
            import json
            return _json_nonempty(json.loads(t))
        except ValueError:
            return True
    if low.startswith("<") and "<body" in low:                       # page HTML vide de GeoServer / MapServer
        body = re.sub(r"<[^>]+>", " ", low[low.index("<body"):])
        return bool(body.strip())
    return True


ID_LIKE = ("cat", "objectid", "id", "fid", "gid", "code", "source", "type", "type_en", "geonunit", "name", "nom")


def info_attributes(text: str) -> dict:
    """Attributs de l'objet interrogé (premier objet), quel que soit le format : tableau {header, rows} du service EFAS,
    GeoJSON (properties), lignes « clé = valeur » ou balises XML. {} si rien."""
    t = (text or "").strip()
    if not t:
        return {}
    if t[0] in "{[":
        try:
            import json
            found = _find_attrs(json.loads(t))
            if found:
                return found
        except ValueError:
            pass
    out = {}
    for m in re.finditer(r"^\s*([A-Za-z_][\w .-]*?)\s*=\s*(.+?)\s*$", t, re.M):
        out.setdefault(m.group(1).strip(), m.group(2).strip().strip("'\""))
    if not out:
        for m in re.finditer(r"<([A-Za-z_][\w:.-]*)[^>/]*>\s*([^<>\s][^<>]*?)\s*</\1>", t):
            out.setdefault(m.group(1), m.group(2))
    return out


def _find_attrs(x) -> dict:
    if isinstance(x, list):
        for i in x:
            r = _find_attrs(i)
            if r:
                return r
    elif isinstance(x, dict):
        data = x.get("data")
        if x.get("type") == "table" and isinstance(data, dict) and data.get("rows"):
            header, row = data.get("header") or [], data["rows"][0]
            return {str(h): v for h, v in zip(header, row)}
        if isinstance(x.get("features"), list) and x["features"]:
            return dict((x["features"][0] or {}).get("properties") or {})
        if x.get("type") not in ("layer_tab", "table", "FeatureCollection") and x:
            return {str(k): v for k, v in x.items() if not isinstance(v, (dict, list))}
    return {}


def _num(v) -> Optional[float]:
    try:
        f = float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def primary_value(attrs: dict) -> tuple[Optional[str], Optional[float]]:
    """(nom de l'attribut, valeur) la plus parlante : période de retour, puis valeur, puis premier numérique hors identifiants."""
    best, best_score = (None, None), 0
    for k, v in (attrs or {}).items():
        n = _num(v)
        low = str(k).lower()
        if n is None or low in ID_LIKE:
            continue
        score = (4 if "return" in low else 0) + (3 if low.startswith("rp") or "_rp" in low else 0) + (2 if "period" in low else 0) \
            + (2 if "protect" in low else 0) + (1 if any(w in low for w in ("value", "depth", "level", "val")) else 0) + 0.1
        if score > best_score:
            best, best_score = (k, n), score
    return best


def _probe_format(base, layer, cells, session, time_value) -> str:
    """Format GetFeatureInfo qui renvoie réellement des données : essayé sur quelques mailles ; à défaut, celui qui répond sans erreur."""
    answered = None
    for c in cells:
        for fmt in FI_FORMATS:
            txt, dg = get_feature_info(base, layer, float(c[0]), float(c[1]), zoom=10, session=session, time_value=time_value,
                                       formats=(fmt,))
            if info_has_feature(txt):
                return fmt
            if answered is None and txt.strip():
                answered = fmt
    return answered or FI_FORMATS[0]


def sample_layer(base: str, layer: str, lats, lons, session=None, time_value: Optional[str] = None, step: float = 0.05,
                 max_cells: int = 300, workers: int = 12, bbox=None) -> tuple[np.ndarray, dict]:
    """Valeur numérique de la couche pour chaque point (NaN si inconnue). Les points sont regroupés par maille de `step` degrés
    pour limiter les appels ; au plus `max_cells` mailles sont interrogées ; les points hors de `bbox` (ouest, sud, est, nord),
    emprise de la couche, ne sont pas interrogés. Retourne (valeurs, bilan) ; le bilan contient aussi `presence` (un objet existe
    à l'endroit du point, même sans valeur numérique) et quelques `exemples` de réponses brutes pour le diagnostic."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    ok = np.isfinite(lats) & np.isfinite(lons)
    out_of = np.zeros(len(lats), dtype=bool)
    if bbox:
        w, s_, e, n = bbox
        out_of = ok & ~((lons >= w) & (lons <= e) & (lats >= s_) & (lats <= n))
        ok = ok & ~out_of
    klat = np.where(ok, np.round(lats / step) * step, np.nan).round(4)
    klon = np.where(ok, np.round(lons / step) * step, np.nan).round(4)
    cells = list(dict.fromkeys(zip(klat[ok], klon[ok])))
    truncated = len(cells) > max_cells
    cells = cells[:max_cells]

    fmt = _probe_format(base, layer, cells[:6], session, time_value)

    def one(c):
        txt, dg = get_feature_info(base, layer, float(c[0]), float(c[1]), zoom=10, session=session, time_value=time_value,
                                   formats=(fmt,))
        attrs = info_attributes(txt)
        present = info_has_feature(txt)
        val = primary_value(attrs)[1] if attrs else None
        if val is None and not attrs:
            val = first_number(txt)
        return c, val, present, txt, dg, attrs

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, cells))
    num = {r[0]: r[1] for r in res}
    pres = {r[0]: r[2] for r in res}
    atts = {r[0]: r[5] for r in res}
    vals = np.full(len(lats), np.nan)
    presence = np.zeros(len(lats), dtype=bool)
    queried = np.zeros(len(lats), dtype=bool)
    attributes: list = [None] * len(lats)
    for i in np.where(ok)[0]:
        key = (float(klat[i]), float(klon[i]))
        v = num.get(key)
        if v is not None:
            vals[i] = v
        presence[i] = bool(pres.get(key, False))
        queried[i] = key in num
        attributes[i] = atts.get(key) or None
    statut = np.array(["hors emprise" if out_of[i] else ("non interrogé (limite)" if not queried[i] else
                                                        ("valeur" if vals[i] == vals[i] else ("objet sans valeur" if presence[i] else "aucun objet")))
                       for i in range(len(lats))], dtype=object)
    statut[~np.isfinite(lats) | ~np.isfinite(lons)] = "coordonnées invalides"
    examples = [{"format": fmt, "lat": c[0], "long": c[1], "réponse": (t or "")[:300], "appels": "; ".join(f"{d['statut']} {d['remarque']}" for d in dg)}
                for c, _, _, t, dg, _a in res[:3]]
    return vals, {"mailles": len(cells), "valeurs_lues": sum(1 for r in res if r[1] is not None),
                  "objets": sum(1 for r in res if r[2]), "tronque": truncated,
                  "hors_emprise": int(out_of.sum()), "presence": presence, "interroge": queried, "statut": statut, "format": fmt, "attributs": attributes, "exemples": examples}


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


# =========================================================================== v3 : contours d'une activation Copernicus EMS
EMS_DETAIL = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"
MAX_GEOJSON_BYTES = 20_000_000
PRODUCT_LABEL = {"REF": "Référence", "FEP": "Première estimation", "DEL": "Délimitation", "GRA": "Évaluation des dégâts",
                 "GRM": "Évaluation des dégâts (suivi)"}


def _find_urls(d: dict) -> dict:
    """URL de GeoJSON et de style SLD d'une couche, quel que soit le nom exact des clés."""
    out = {"geojson": "", "sld": ""}
    for k, v in d.items():
        if not isinstance(v, str) or not v.lower().startswith("http"):
            continue
        low = v.lower().split("?")[0]
        if low.endswith((".geojson", ".json")) and not out["geojson"]:
            out["geojson"] = v
        elif low.endswith(".sld") and not out["sld"]:
            out["sld"] = v
        elif "geojson" in str(k).lower() and not out["geojson"]:
            out["geojson"] = v
        elif ("sld" in str(k).lower() or "style" in str(k).lower()) and not out["sld"]:
            out["sld"] = v
    return out


def iter_layers(detail) -> list[dict]:
    """Couches vectorielles d'une activation (réponse de public-activations/?code=…), en parcourant toute la structure
    (AOI > produits > couches) : [{aoi, produit, type, numero, suivi, nom, geojson, sld}]."""
    out: list[dict] = []

    def walk(x, ctx):
        if isinstance(x, list):
            for i in x:
                walk(i, ctx)
        elif isinstance(x, dict):
            c = dict(ctx)
            if "number" in x and ("products" in x or "aoi" in str(x.get("name", "")).lower() or x.get("name")):
                c["aoi"] = str(x.get("name") or f"AOI {x.get('number')}")
            if x.get("type") in PRODUCT_LABEL or "monitoringNumber" in x:
                c.update({"type": str(x.get("type") or ""), "numero": x.get("number"),
                          "suivi": x.get("monitoringNumber")})
            urls = _find_urls(x)
            if urls["geojson"]:
                out.append({"aoi": c.get("aoi", ""), "type": c.get("type", ""), "numero": c.get("numero"), "suivi": c.get("suivi"),
                            "nom": str(x.get("name") or x.get("title") or x.get("layer") or x.get("fileName")
                                       or urls["geojson"].rsplit("/", 1)[-1]),
                            "geojson": urls["geojson"], "sld": urls["sld"]})
                return
            for v in x.values():
                if isinstance(v, (list, dict)):
                    walk(v, c)

    walk(detail, {})
    seen, uniq = set(), []
    for l in out:
        if l["geojson"] not in seen:
            seen.add(l["geojson"])
            uniq.append(l)
    return uniq


def fetch_activation_detail(code: str, session=None) -> tuple[Optional[dict], list[dict]]:
    diag = []
    try:
        r = _get(session, EMS_DETAIL, {"code": code})
    except requests.RequestException as e:
        return None, [_diag(EMS_DETAIL, "erreur", str(e)[:150])]
    if r.status_code != 200:
        return None, [_diag(EMS_DETAIL, r.status_code, "réponse inattendue")]
    try:
        data = r.json()
    except ValueError:
        return None, [_diag(EMS_DETAIL, 200, "réponse non JSON")]
    res = data.get("results") if isinstance(data, dict) else data
    item = res[0] if isinstance(res, list) and res else (data if isinstance(data, dict) else None)
    diag.append(_diag(EMS_DETAIL, 200, "activation lue" if item else "activation introuvable"))
    return item, diag


def fetch_geojson(url: str, session=None) -> tuple[Optional[dict], str]:
    """GeoJSON d'une couche (taille limitée). Retourne (données, message d'erreur éventuel)."""
    try:
        r = _get(session, url)
    except requests.RequestException as e:
        return None, f"erreur réseau : {str(e)[:120]}"
    if r.status_code != 200:
        return None, f"statut {r.status_code}"
    size = len(getattr(r, "content", b"") or b"") or len(getattr(r, "text", "") or "")
    if size > MAX_GEOJSON_BYTES:
        return None, f"fichier trop volumineux ({size / 1e6:.0f} Mo, limite {MAX_GEOJSON_BYTES / 1e6:.0f} Mo)"
    try:
        data = r.json()
    except ValueError:
        return None, "réponse non JSON"
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        return None, "pas de « features »"
    return data, ""


# --------------------------------------------------------------------------- styles SLD
def parse_sld(xml_text: str) -> list[dict]:
    """Règles de style d'un fichier SLD : [{titre, champ, valeur, fill, stroke, fill_opacity, stroke_width}] (champ/valeur : filtre
    d'égalité éventuel, None pour la règle par défaut)."""
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        return []
    tag = lambda e: e.tag.split("}")[-1]
    rules = []
    for rule in (e for e in root.iter() if tag(e) == "Rule"):
        r = {"titre": "", "champ": None, "valeur": None, "fill": None, "stroke": None, "fill_opacity": None, "stroke_width": None}
        for ch in rule:
            if tag(ch) in ("Title", "Name") and (ch.text or "").strip() and not r["titre"]:
                r["titre"] = ch.text.strip()
        for f in rule.iter():
            if tag(f) == "PropertyIsEqualTo":
                pn = next((x.text for x in f.iter() if tag(x) == "PropertyName"), None)
                lit = next((x.text for x in f.iter() if tag(x) == "Literal"), None)
                if pn and lit is not None:
                    r["champ"], r["valeur"] = pn.strip(), lit.strip()
                break
        for sym in (e for e in rule.iter() if tag(e).endswith("Symbolizer")):
            for part in sym.iter():
                if tag(part) not in ("Fill", "Stroke"):
                    continue
                params = {p.get("name"): (p.text or "").strip() for p in part.iter() if tag(p) in ("CssParameter", "SvgParameter")}
                if tag(part) == "Fill":
                    r["fill"] = r["fill"] or params.get("fill")
                    r["fill_opacity"] = r["fill_opacity"] if r["fill_opacity"] is not None else _num(params.get("fill-opacity"))
                else:
                    r["stroke"] = r["stroke"] or params.get("stroke")
                    r["stroke_width"] = r["stroke_width"] if r["stroke_width"] is not None else _num(params.get("stroke-width"))
        if r["fill"] or r["stroke"]:
            rules.append(r)
    return rules


DEFAULT_COLORS = (("observed", "#1f6fb5"), ("flood", "#1f6fb5"), ("water", "#1f6fb5"), ("burnt", "#d7191c"), ("fire", "#d7191c"),
                  ("aoi", "#555555"), ("built", "#8c6d31"), ("damage", "#e08214"), ("grade", "#e08214"))


def default_rule(name: str) -> dict:
    low = name.lower()
    color = next((c for k, c in DEFAULT_COLORS if k in low), "#d7191c")
    return {"titre": name, "champ": None, "valeur": None, "fill": color if "aoi" not in low else None, "stroke": color,
            "fill_opacity": 0.45, "stroke_width": 1.5}


def style_for(props: dict, rules: list[dict]) -> dict:
    """Style Leaflet d'une entité : première règle dont le filtre correspond, sinon règle par défaut, sinon rouge."""
    chosen = None
    for r in rules:
        if r["champ"] is None:
            chosen = chosen or r
        elif str((props or {}).get(r["champ"])) == str(r["valeur"]):
            chosen = r
            break
    r = chosen or (rules[0] if rules else default_rule(""))
    return {"color": r["stroke"] or r["fill"] or "#d7191c", "weight": r["stroke_width"] or 1.2,
            "fillColor": r["fill"] or "#000000", "fillOpacity": (r["fill_opacity"] if r["fill_opacity"] is not None else 0.5) if r["fill"] else 0.0}


def fetch_text(url: str, session=None) -> str:
    """Texte d'un fichier (style SLD…) ; chaîne vide en cas d'échec."""
    try:
        r = _get(session, url)
        return r.text if r.status_code == 200 else ""
    except requests.RequestException:
        return ""


def geojson_bounds(gj: Optional[dict]) -> Optional[tuple[float, float, float, float]]:
    """(sud, ouest, nord, est) de toutes les coordonnées d'un GeoJSON ; None s'il n'y en a pas."""
    pts: list = []

    def walk(c):
        if isinstance(c, (list, tuple)) and c:
            if isinstance(c[0], (int, float)) and len(c) >= 2:
                pts.append((float(c[1]), float(c[0])))
            else:
                for i in c:
                    walk(i)

    for f in (gj or {}).get("features") or []:
        walk((f.get("geometry") or {}).get("coordinates"))
    if not pts:
        return None
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    return min(lats), min(lons), max(lats), max(lons)


def legend_items(rules: list[dict], layer_name: str) -> list[dict]:
    """Éléments de légende [{titre, couleur, contour}] d'une couche à partir de ses règles de style (style par défaut sinon)."""
    rules = rules or [default_rule(layer_name)]
    out = []
    for r in rules:
        t = r["titre"] or (f"{r['champ']} = {r['valeur']}" if r["champ"] else layer_name)
        out.append({"titre": t, "couleur": r["fill"] or "transparent", "contour": r["stroke"] or r["fill"] or "#555555"})
    return out


def fetch_style_legend(base: str, layer: str, session=None) -> tuple[list[dict], list[dict]]:
    """Légende de secours quand le service ne publie pas d'image : style (SLD) de la couche demandé par GetStyles, converti en
    éléments de légende [{titre, couleur, contour}]. Retourne (éléments, diagnostic)."""
    diag = []
    for ver in ("1.1.0", "1.3.0"):
        sep = "&" if "?" in base else "?"
        u = base + sep + urlencode({"service": "WMS", "version": ver, "request": "GetStyles", "layers": layer})
        try:
            r = _get(session, u)
        except requests.RequestException as e:
            diag.append(_diag(u, "erreur", str(e)[:150]))
            continue
        if r.status_code != 200:
            diag.append(_diag(u, r.status_code, "GetStyles refusé"))
            continue
        rules = parse_sld(r.text)
        diag.append(_diag(u, 200, f"{len(rules)} règle(s) de style"))
        if rules:
            return legend_items(rules, layer), diag
    return [], diag


def cross_with_layer(df: pd.DataFrame, lat_c: str, lon_c: str, gj: Optional[dict], name: str, snap_km: float = 0.0,
                     max_props: int = 8) -> dict:
    """Croise des points avec les polygones d'une couche : colonnes `<name>.statut` (« Dans la zone » / « À moins de X km » /
    « Hors zone »), `<name>.distance_km` et les attributs de l'objet qui contient (ou est le plus proche du) point."""
    from .polygon_index import PolygonIndex
    lat = pd.to_numeric(df[lat_c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()
    lon = pd.to_numeric(df[lon_c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy()
    n = len(df)
    cols: dict = {f"{name}.statut": ["Hors zone"] * n, f"{name}.distance_km": [None] * n}
    feats = (gj or {}).get("features") or []
    polys = [rings_from_feature(f) for f in feats]
    keep = [i for i, r in enumerate(polys) if r]
    if not keep:
        return cols
    idx = PolygonIndex([polys[i] for i in keep])
    ids, methods, dist = idx.locate(lon, lat, snap_km=snap_km)
    props = [feats[keep[i]].get("properties") or {} for i in range(len(keep))]
    keys = []
    for p_ in props:
        for k in p_:
            if k not in keys and not isinstance(p_[k], (dict, list)):
                keys.append(k)
    for k in keys[:max_props]:
        cols[f"{name}.{k}"] = [props[i].get(k) if i >= 0 else None for i in ids]
    st_, dk = [], []
    for m, d in zip(methods, dist):
        if m == "within":
            st_.append("Dans la zone")
            dk.append(0.0)
        elif m == "nearest":
            st_.append(f"À moins de {snap_km:g} km")
            dk.append(round(float(d), 2))
        else:
            st_.append("Hors zone")
            dk.append(None)
    cols[f"{name}.statut"], cols[f"{name}.distance_km"] = st_, dk
    return cols
