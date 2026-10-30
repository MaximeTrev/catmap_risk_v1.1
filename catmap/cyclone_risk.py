"""Croisement d'une liste de points (CSV) avec la trajectoire prévue des cyclones actifs.

Définitions (à connaître pour interpréter les colonnes ajoutées) :

* « Sur la trajectoire » : le point se trouve à moins de ``radius_km`` de la ligne de trajectoire
  prévue (position actuelle comprise). Le rayon est réglable ; ce n'est PAS le rayon réel des
  vents, qui varie d'un cyclone à l'autre.
* ``impact`` : première échéance de prévision (12, 24, 36… heures) à laquelle le centre du
  cyclone est entré dans le rayon autour du point (le premier pas de prévision ≥ à cet instant ;
  « 12 » = dans les 12 prochaines heures, y compris si le cyclone est déjà là). 0 = hors trajectoire.
* ``max_sustained_winds`` / ``max_wind_gusts`` (km/h) : maximum entre le point de prévision de
  l'échéance d'impact et le point précédent (le plus fort des deux). Source : NHC/JTWC.
* ``total_precipitation`` (mm) : cumul horaire prévu AU POINT (Open-Meteo, modèle global) depuis
  maintenant jusqu'à 12 h après l'échéance d'impact. Hors trajectoire : None.
* Plusieurs cyclones : celui qui atteint le point le plus tôt est retenu (à égalité, le plus fort).
"""
from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .cyclones import KT_TO_KMH, CycloneError, Storm, _get_json, OPEN_METEO_URL
from .inputs import InputError, _decode, _sniff_delimiter

LAT_NAMES = ("lat", "latitude", "lats", "y", "lat_deg")
LON_NAMES = ("lon", "long", "lng", "longitude", "lons", "x", "lon_deg")
NEW_COLUMNS = ["impact", "max_sustained_winds", "max_wind_gusts", "total_precipitation", "cyclone", "distance_trajectoire_km"]
EARTH_KM = 6371.0088
SUBSTEPS = 24          # points interpolés par segment de prévision (≈ 10 à 20 km)
PRECIP_BATCH = 50


# --------------------------------------------------------------------------- lecture
def _to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def read_points_csv(data: bytes, filename: str = "", reserved_check: bool = True) -> tuple[pd.DataFrame, str, str]:
    """Lit un CSV (séparateur détecté, virgule décimale tolérée) ou un Excel (feuille « sites » si elle existe, sinon la
    première : c'est l'export de la page Géolocalisation). Retourne (df, colonne_lat, colonne_lon)."""
    if filename.lower().endswith((".xlsx", ".xlsm")):
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
        except Exception as e:  # noqa: BLE001
            raise InputError(f"Lecture Excel impossible : {e}") from e
        if not sheets:
            raise InputError("Classeur vide.")
        df = sheets.get("sites", next(iter(sheets.values()))).fillna("")
    else:
        text = _decode(data)
        if not text.strip():
            raise InputError("Fichier vide.")
        delim = _sniff_delimiter(text)
        try:
            df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            raise InputError(f"Lecture CSV impossible : {e}") from e
    if df.empty:
        raise InputError("Aucune ligne dans le fichier.")
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    lat_c = next((low[n] for n in LAT_NAMES if n in low), None)
    lon_c = next((low[n] for n in LON_NAMES if n in low), None)
    if not lat_c or not lon_c:
        raise InputError("Colonnes de coordonnées introuvables. Nommez-les « lat » et « long » "
                         "(ou latitude/longitude, lon, lng). Colonnes lues : " + ", ".join(df.columns[:12]))
    clash = [c for c in NEW_COLUMNS if c in df.columns] if reserved_check else []
    if clash:
        raise InputError("Le fichier contient déjà des colonnes réservées : " + ", ".join(clash)
                         + ". Retirez-les ou renommez-les avant l'analyse.")
    return df, lat_c, lon_c


# --------------------------------------------------------------------------- géométrie
def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def densify(storm: Storm) -> np.ndarray:
    """Trajectoire prévue échantillonnée finement : tableau (n, 3) = tau, lat, lon (interpolation linéaire)."""
    pts = sorted(storm.forecast, key=lambda p: p.tau)
    rows = []
    for a, b in zip(pts, pts[1:]):
        for k in range(SUBSTEPS):
            f = k / SUBSTEPS
            rows.append((a.tau + f * (b.tau - a.tau), a.lat + f * (b.lat - a.lat), a.lon + f * (b.lon - a.lon)))
    if pts:
        rows.append((pts[-1].tau, pts[-1].lat, pts[-1].lon))
    return np.array(rows, dtype=float)


def _storm_hits(storm: Storm, lat: np.ndarray, lon: np.ndarray, radius_km: float):
    """Pour chaque point : (tau de première entrée dans le rayon ou NaN, distance minimale à la trajectoire en km)."""
    track = densify(storm)
    n = len(lat)
    first = np.full(n, np.nan)
    dmin = np.full(n, np.inf)
    if len(track) == 0:
        return first, dmin
    for start in range(0, n, 20000):       # blocs : limite la mémoire (n × échantillons)
        sl = slice(start, start + 20000)
        d = _haversine_km(lat[sl, None], lon[sl, None], track[None, :, 1], track[None, :, 2])
        dmin[sl] = d.min(axis=1)
        inside = d <= radius_km
        any_in = inside.any(axis=1)
        idx = inside.argmax(axis=1)
        block_first = np.where(any_in, track[idx, 0], np.nan)
        first[sl] = block_first
    return first, dmin


# --------------------------------------------------------------------------- précipitations au point
def site_precipitation(points: Sequence[tuple[float, float, datetime, datetime]], *, session=None) -> list[Optional[float]]:
    """Cumul horaire Open-Meteo par point (lat, lon, début, fin). Valeur None si indisponible ou couverture insuffisante."""
    out: list[Optional[float]] = [None] * len(points)
    for s in range(0, len(points), PRECIP_BATCH):
        chunk = points[s:s + PRECIP_BATCH]
        params = {"latitude": ",".join(f"{p[0]:.3f}" for p in chunk),
                  "longitude": ",".join(f"{((p[1] + 180) % 360) - 180:.3f}" for p in chunk),
                  "hourly": "precipitation", "past_days": 2, "forecast_days": 10, "timezone": "UTC"}
        try:
            data = _get_json(OPEN_METEO_URL, params, session=session)
        except CycloneError:
            continue
        results = data if isinstance(data, list) else [data]
        for i, (p, res) in enumerate(zip(chunk, results)):
            hourly = (res or {}).get("hourly") or {}
            times, vals = hourly.get("time") or [], hourly.get("precipitation") or []
            total, n_h = 0.0, 0
            for t, v in zip(times, vals):
                try:
                    ts = datetime.strptime(t, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if p[2] <= ts < p[3] and v is not None:
                    total += float(v)
                    n_h += 1
            expected = max(1, int((p[3] - p[2]).total_seconds() // 3600))
            if n_h >= 0.8 * expected:
                out[s + i] = round(total, 1)
    return out


# --------------------------------------------------------------------------- croisement
def assess_points(df: pd.DataFrame, lat_col: str, lon_col: str, storms: Sequence[Storm], *, radius_km: float = 150.0,
                  precip_fetcher: Optional[Callable] = None, now: Optional[datetime] = None) -> pd.DataFrame:
    """Retourne une copie de ``df`` avec les colonnes NEW_COLUMNS. Les points sans coordonnées valides restent
    dans le résultat avec impact = 0 et des valeurs None."""
    now = now or datetime.now(timezone.utc)
    lat = _to_float(df[lat_col]).to_numpy()
    lon = _to_float(df[lon_col]).to_numpy()
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    n = len(df)
    best_first = np.full(n, np.inf)
    best_wind = np.full(n, -1.0)
    best = [None] * n            # (storm, impact_tau, dist)

    vl, vo = lat[valid], lon[valid]
    vidx = np.flatnonzero(valid)
    for storm in storms:
        if storm.current is None or len(storm.forecast) < 1:
            continue
        steps = sorted({p.tau for p in storm.forecast if p.tau > 0})
        if not steps:       # pas de prévision : seule la position actuelle (impact = 12 h par convention)
            steps = [12]
        first, dmin = _storm_hits(storm, vl, vo, radius_km)
        for j in np.flatnonzero(np.isfinite(first)):
            t = first[j]
            impact = next((s for s in steps if s >= t - 1e-9), steps[-1])
            i = vidx[j]
            w = max((p.wind_kt or 0) for p in storm.forecast if p.tau == impact) if any(p.tau == impact for p in storm.forecast) else (storm.current.wind_kt or 0)
            if impact < best_first[i] or (impact == best_first[i] and w > best_wind[i]):
                best_first[i], best_wind[i] = impact, w
                best[i] = (storm, impact, float(dmin[j]))

    impact_col = np.zeros(n, dtype=int)
    wind_col: list = [None] * n
    gust_col: list = [None] * n
    cyc_col: list = [None] * n
    dist_col: list = [None] * n
    windows = []   # (i, lat, lon, start, end)
    for i, b in enumerate(best):
        if b is None:
            continue
        storm, impact, dist = b
        pts = sorted(storm.forecast, key=lambda p: p.tau)
        at = [p for p in pts if p.tau == impact] or [storm.current]
        prev = [p for p in pts if p.tau < impact][-1:]
        pair = prev + at
        winds = [p.wind_kt for p in pair if p.wind_kt]
        gusts = [p.gust_kt for p in pair if p.gust_kt]
        impact_col[i] = impact
        wind_col[i] = round(max(winds) * KT_TO_KMH) if winds else None
        gust_col[i] = round(max(gusts) * KT_TO_KMH) if gusts else None
        cyc_col[i] = storm.name
        dist_col[i] = round(dist, 1)
        base = storm.advisory_time or now
        windows.append((i, lat[i], lon[i], now.replace(minute=0, second=0, microsecond=0), base + timedelta(hours=impact + 12)))

    precip_col: list = [None] * n
    if windows:
        fetch = precip_fetcher or site_precipitation
        # points identiques (≈ 1 km) : une seule requête
        key = lambda w: (round(w[1], 2), round(w[2], 2), w[3], w[4])  # noqa: E731
        uniq: dict = {}
        for w in windows:
            uniq.setdefault(key(w), (w[1], w[2], w[3], w[4]))
        keys = list(uniq)
        vals = fetch([uniq[k] for k in keys])
        lookup = dict(zip(keys, vals))
        for w in windows:
            precip_col[w[0]] = lookup.get(key(w))

    out = df.copy()
    out["impact"] = impact_col
    out["max_sustained_winds"] = wind_col
    out["max_wind_gusts"] = gust_col
    out["total_precipitation"] = precip_col
    out["cyclone"] = cyc_col
    out["distance_trajectoire_km"] = dist_col
    return out
