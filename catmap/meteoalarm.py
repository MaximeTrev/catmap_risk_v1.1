"""Alertes météo européennes agrégées par MeteoAlarm (flux publics par pays, format CAP 1.2).

Source : ``https://feeds.meteoalarm.org/api/v1/warnings/feeds-<pays>`` (un flux par service météo national).
Licence des données : CC BY 4.0 (MeteoAlarm / EUMETNET) ; ces flux sont mis en cache 3 minutes côté MeteoAlarm.

Le format exact des réponses n'a pas pu être vérifié depuis l'environnement de développement : l'analyse est donc
volontairement tolérante (alerte à la racine ou sous ``alert``, ``info`` en liste ou objet, paramètres en liste
``valueName/value`` ou en dictionnaire, polygones CAP « lat,lon lat,lon » ou GeoJSON). Une réponse non reconnue
produit un diagnostic explicite et jamais d'exception.

Niveaux MeteoAlarm (``awareness_level``, ex. « 2; yellow; Moderate ») : 1 vert, 2 jaune, 3 orange, 4 rouge.
Types (``awareness_type``, ex. « 1; Wind ») : voir ``TYPES``.
Seuls les pays dont le flux fournit des polygones (ex. France, Norvège, Suisse, Suède, Royaume-Uni) peuvent être
dessinés et croisés avec des points ; les autres n'indiquent que des codes de zones (EMMA_ID) sans contour.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import requests

from .polygon_index import PolygonIndex

FEED_URL = "https://feeds.meteoalarm.org/api/v1/warnings/feeds-{slug}"

# slug du flux -> (libellé, ISO2, centre approximatif lat/lon)
COUNTRIES = {
    "austria": ("Autriche", "AT", (47.6, 14.1)), "belgium": ("Belgique", "BE", (50.6, 4.7)),
    "bosnia-herzegovina": ("Bosnie-Herzégovine", "BA", (44.2, 17.8)), "bulgaria": ("Bulgarie", "BG", (42.7, 25.2)),
    "croatia": ("Croatie", "HR", (45.1, 15.5)), "cyprus": ("Chypre", "CY", (35.0, 33.2)),
    "czechia": ("Tchéquie", "CZ", (49.8, 15.5)), "denmark": ("Danemark", "DK", (56.0, 9.5)),
    "estonia": ("Estonie", "EE", (58.7, 25.5)), "finland": ("Finlande", "FI", (64.0, 26.0)),
    "france": ("France", "FR", (46.6, 2.5)), "germany": ("Allemagne", "DE", (51.1, 10.4)),
    "greece": ("Grèce", "GR", (39.0, 22.5)), "hungary": ("Hongrie", "HU", (47.2, 19.4)),
    "iceland": ("Islande", "IS", (64.9, -18.5)), "ireland": ("Irlande", "IE", (53.2, -8.0)),
    "israel": ("Israël", "IL", (31.5, 34.9)), "italy": ("Italie", "IT", (42.8, 12.6)),
    "latvia": ("Lettonie", "LV", (56.9, 24.9)), "lithuania": ("Lituanie", "LT", (55.3, 23.9)),
    "luxembourg": ("Luxembourg", "LU", (49.8, 6.1)), "malta": ("Malte", "MT", (35.9, 14.4)),
    "moldova": ("Moldavie", "MD", (47.2, 28.5)), "montenegro": ("Monténégro", "ME", (42.8, 19.3)),
    "netherlands": ("Pays-Bas", "NL", (52.2, 5.5)), "north-macedonia": ("Macédoine du Nord", "MK", (41.6, 21.7)),
    "norway": ("Norvège", "NO", (64.5, 11.5)), "poland": ("Pologne", "PL", (52.0, 19.4)),
    "portugal": ("Portugal", "PT", (39.6, -8.0)), "romania": ("Roumanie", "RO", (45.9, 24.9)),
    "serbia": ("Serbie", "RS", (44.0, 20.8)), "slovakia": ("Slovaquie", "SK", (48.7, 19.5)),
    "slovenia": ("Slovénie", "SI", (46.1, 14.8)), "spain": ("Espagne", "ES", (40.2, -3.7)),
    "sweden": ("Suède", "SE", (62.0, 15.0)), "switzerland": ("Suisse", "CH", (46.8, 8.2)),
    "ukraine": ("Ukraine", "UA", (49.0, 31.4)), "united-kingdom": ("Royaume-Uni", "GB", (54.0, -2.5)),
}
LEVELS = {1: "Vert", 2: "Jaune", 3: "Orange", 4: "Rouge"}
LEVEL_HEX = {1: "#2e9e4f", 2: "#f2c200", 3: "#ef7d00", 4: "#d1202f"}
TYPES = {1: "Vent", 2: "Neige / verglas", 3: "Orages", 4: "Brouillard", 5: "Fortes chaleurs", 6: "Grand froid",
         7: "Événement côtier", 8: "Feu de forêt", 9: "Avalanches", 10: "Fortes pluies", 11: "Inondations",
         12: "Pluie-inondation"}
SEVERITY_TO_LEVEL = {"minor": 1, "moderate": 2, "severe": 3, "extreme": 4}
COLUMNS = ["pays", "pays_iso2", "identifiant", "emetteur", "evenement", "niveau_id", "niveau", "type_id", "type",
           "debut", "fin", "emis", "urgence", "severite", "certitude", "zones", "codes_zones", "titre", "description",
           "consigne", "langues", "avec_contour"]


class MeteoAlarmError(RuntimeError):
    pass


# --------------------------------------------------------------------------- HTTP
def fetch_country(slug: str, *, session=None, url_template: str = FEED_URL, timeout: float = 25.0) -> Any:
    s = session or requests
    url = url_template.format(slug=slug)
    try:
        r = s.get(url, headers={"Accept": "application/json", "User-Agent": "CatMap/3.3"}, timeout=timeout)
    except requests.RequestException as e:
        raise MeteoAlarmError(f"flux injoignable ({type(e).__name__})") from e
    if r.status_code == 404:
        raise MeteoAlarmError("flux introuvable (HTTP 404)")
    if r.status_code in (401, 403):
        raise MeteoAlarmError(f"accès refusé (HTTP {r.status_code})")
    if r.status_code != 200:
        raise MeteoAlarmError(f"HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError as e:
        raise MeteoAlarmError("réponse non JSON (format de flux différent de celui attendu)") from e


# --------------------------------------------------------------------------- analyse
def _dt(v) -> Optional[datetime]:
    if not v:
        return None
    s = str(v).strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _params(info: dict) -> dict:
    out = {}
    p = info.get("parameter", info.get("parameters"))
    if isinstance(p, dict):
        return {str(k): v for k, v in p.items()}
    for it in _as_list(p):
        if isinstance(it, dict) and "valueName" in it:
            out[str(it["valueName"])] = it.get("value")
    return out


def _lead_int(text) -> Optional[int]:
    m = re.match(r"\s*(\d+)", str(text or ""))
    return int(m.group(1)) if m else None


def parse_polygon(poly) -> Optional[np.ndarray]:
    """Anneau CAP « lat,lon lat,lon … » (ou liste de paires) -> tableau (n, 2) en (lon, lat)."""
    pts = []
    if isinstance(poly, str):
        for tok in poly.split():
            try:
                a, b = tok.split(",")[:2]
                pts.append((float(b), float(a)))
            except ValueError:
                continue
    elif isinstance(poly, (list, tuple)):
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append((float(p[1]), float(p[0])))      # CAP en liste : (lat, lon)
                except (TypeError, ValueError):
                    continue
    return np.array(pts, dtype=float) if len(pts) >= 3 else None


def _geojson_rings(geom: dict) -> list[np.ndarray]:
    t, c = geom.get("type"), geom.get("coordinates") or []
    polys = c if t == "MultiPolygon" else [c] if t == "Polygon" else []
    return [np.asarray(r, dtype=float)[:, :2] for poly in polys for r in poly if len(r) >= 3]


def _pick_language(infos: list[dict]) -> dict:
    for i in infos:
        if str(i.get("language", "")).lower().startswith("en"):
            return i
    return infos[0]


def parse_feed(payload: Any, slug: str, *, now: Optional[datetime] = None, only_active: bool = True
               ) -> tuple[pd.DataFrame, dict[int, list[np.ndarray]]]:
    """Retourne (table des alertes, {index de ligne: anneaux (lon, lat)}). Lève MeteoAlarmError si le format est inconnu."""
    now = now or datetime.now(timezone.utc)
    label, iso2, _ = COUNTRIES.get(slug, (slug, "", (0, 0)))
    if isinstance(payload, dict):
        items = payload.get("warnings", payload.get("alerts", payload.get("features")))
        if items is None and ("alert" in payload or "info" in payload):
            items = [payload]
    else:
        items = payload
    if not isinstance(items, list):
        keys = sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        raise MeteoAlarmError(f"format non reconnu (clés de premier niveau : {keys})")
    rows, rings = [], {}
    for it in items:
        if not isinstance(it, dict):
            continue
        alert = it.get("alert") if isinstance(it.get("alert"), dict) else it
        if not isinstance(alert, dict):
            continue
        if str(alert.get("status", "Actual")).lower() in ("test", "exercise", "draft", "system"):
            continue
        if str(alert.get("msgType", "Alert")).lower() == "cancel":
            continue
        infos = [i for i in _as_list(alert.get("info")) if isinstance(i, dict)]
        if not infos:
            continue
        groups: dict[tuple, list[dict]] = {}
        for i in infos:                       # une info par langue et par phénomène : on regroupe par phénomène
            p = _params(i)
            key = (_lead_int(p.get("awareness_type")), _lead_int(p.get("awareness_level")), str(i.get("onset", "")),
                   str(i.get("event", "")) if p.get("awareness_type") is None else "")
            groups.setdefault(key, []).append(i)
        for (type_id, level_id, _, _), glist in groups.items():
            info = _pick_language(glist)
            p = _params(info)
            if level_id is None:
                level_id = SEVERITY_TO_LEVEL.get(str(info.get("severity", "")).lower())
            if type_id is None:
                type_id = _lead_int(p.get("awareness_type"))
            tlabel = TYPES.get(type_id) if type_id else None
            if tlabel is None:
                tail = str(p.get("awareness_type", "")).split(";", 1)
                tlabel = tail[1].strip() if len(tail) > 1 and tail[1].strip() else (info.get("event") or "Autre")
            onset = _dt(info.get("onset") or info.get("effective") or alert.get("sent"))
            expires = _dt(info.get("expires"))
            if only_active and expires is not None and expires < now:
                continue
            areas = [a for a in _as_list(info.get("area")) if isinstance(a, dict)]
            ring_list: list[np.ndarray] = []
            codes = []
            for a in areas:
                for poly in _as_list(a.get("polygon")):
                    r = parse_polygon(poly)
                    if r is not None:
                        ring_list.append(r)
                if isinstance(a.get("geometry"), dict):
                    ring_list.extend(_geojson_rings(a["geometry"]))
                for g in _as_list(a.get("geocode")):
                    if isinstance(g, dict) and g.get("value"):
                        codes.append(str(g["value"]))
            idx = len(rows)
            if ring_list:
                rings[idx] = ring_list
            rows.append({
                "pays": label, "pays_iso2": iso2, "identifiant": str(alert.get("identifier", "")),
                "emetteur": str(alert.get("senderName") or alert.get("sender") or ""),
                "evenement": str(info.get("event", "")), "niveau_id": level_id, "niveau": LEVELS.get(level_id, "n/d"),
                "type_id": type_id, "type": tlabel, "debut": onset, "fin": expires, "emis": _dt(alert.get("sent")),
                "urgence": str(info.get("urgency", "")), "severite": str(info.get("severity", "")),
                "certitude": str(info.get("certainty", "")),
                "zones": "; ".join(str(a.get("areaDesc", "")) for a in areas if a.get("areaDesc")),
                "codes_zones": ", ".join(sorted(set(codes))),
                "titre": str(info.get("headline", "")), "description": str(info.get("description", "")),
                "consigne": str(info.get("instruction", "")),
                "langues": ", ".join(sorted({str(i.get("language", "")) for i in glist if i.get("language")})),
                "avec_contour": bool(ring_list),
            })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["niveau_id"] = pd.to_numeric(df["niveau_id"], errors="coerce")
    return df, rings


def fetch_all(slugs: Iterable[str], *, session=None, url_template: str = FEED_URL, workers: int = 8,
              now: Optional[datetime] = None) -> tuple[pd.DataFrame, dict[int, list[np.ndarray]], dict[str, str]]:
    """Interroge les pays en parallèle. Retourne (alertes, contours indexés sur les lignes, {pays: erreur})."""
    slugs = list(slugs)
    errors: dict[str, str] = {}

    def one(slug):
        try:
            return slug, parse_feed(fetch_country(slug, session=session, url_template=url_template), slug, now=now), None
        except MeteoAlarmError as e:
            return slug, None, str(e)

    frames, all_rings, offset = [], {}, 0
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(slugs) or 1))) as ex:
        results = list(ex.map(one, slugs))
    for slug, res, err in results:
        label = COUNTRIES.get(slug, (slug,))[0]
        if err:
            errors[label] = err
            continue
        df, rings = res
        for k, v in rings.items():
            all_rings[offset + k] = v
        offset += len(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLUMNS), {}, errors
    return pd.concat(frames, ignore_index=True), all_rings, errors


def worst_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Niveau maximal, nombre d'alertes et types par pays."""
    if df.empty:
        return pd.DataFrame(columns=["pays", "niveau_id", "niveau", "alertes", "types"])
    out = []
    for pays, g in df.groupby("pays"):
        mx = g["niveau_id"].max()
        top = g[g["niveau_id"] >= 2]
        out.append({"pays": pays, "niveau_id": mx, "niveau": LEVELS.get(int(mx), "n/d") if pd.notna(mx) else "n/d",
                    "alertes": len(g), "types": ", ".join(sorted(set(top["type"]))) if len(top) else ""})
    return pd.DataFrame(out).sort_values(["niveau_id", "pays"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- croisement avec des points
def expose_points(lat: np.ndarray, lon: np.ndarray, alerts: pd.DataFrame, rings: dict[int, list[np.ndarray]],
                  min_level: int = 2, now: Optional[datetime] = None) -> pd.DataFrame:
    """Pour chaque point : niveau maximal parmi les alertes dont un contour le contient, avec types, dates et zones.
    Les alertes sans contour dans le flux (pays à codes de zones seulement) ne peuvent pas être croisées."""
    n = len(lat)
    out = pd.DataFrame({"europe_niveau_id": [None] * n, "europe_niveau": [None] * n, "europe_types": [None] * n,
                        "europe_debut": [None] * n, "europe_fin": [None] * n, "europe_zones": [None] * n,
                        "europe_pays": [None] * n})
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    hits: dict[int, list[int]] = {}
    sel = [i for i in rings if i < len(alerts) and (alerts.iloc[i]["niveau_id"] or 0) >= min_level]
    for ai in sel:
        idx = PolygonIndex([rings[ai]])
        x0, y0, x1, y1 = idx._bbox[0]
        cand = np.flatnonzero(ok & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
        if len(cand) == 0:
            continue
        inside = idx._inside(lon[cand], lat[cand], 0)
        for p in cand[inside]:
            hits.setdefault(int(p), []).append(ai)
    for p, ais in hits.items():
        g = alerts.iloc[ais]
        mx = int(g["niveau_id"].max())
        worst = g[g["niveau_id"] == mx]
        out.at[p, "europe_niveau_id"] = mx
        out.at[p, "europe_niveau"] = LEVELS.get(mx, "n/d")
        out.at[p, "europe_types"] = ", ".join(sorted(set(f"{r.type} ({r.niveau})" for r in g.itertuples())))
        deb = [d for d in worst["debut"] if d is not None and pd.notna(d)]
        fin = [d for d in worst["fin"] if d is not None and pd.notna(d)]
        out.at[p, "europe_debut"] = min(deb).strftime("%Y-%m-%d %H:%M UTC") if deb else None
        out.at[p, "europe_fin"] = max(fin).strftime("%Y-%m-%d %H:%M UTC") if fin else None
        out.at[p, "europe_zones"] = "; ".join(sorted(set(z for z in worst["zones"] if z)))[:300] or None
        out.at[p, "europe_pays"] = ", ".join(sorted(set(g["pays"])))
    return out
