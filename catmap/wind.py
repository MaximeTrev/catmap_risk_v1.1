"""Vent et tempêtes à partir des données Copernicus (CDS) : rafales par période de retour et analyse événementielle.

Sources (gratuites, CC-BY ; compte et clé d'API Copernicus requis, clé dans les secrets Streamlit ``CDSAPI_KEY``) :

* ``reanalysis-era5-single-levels-timeseries`` : ERA5 horaire par point (grille 0,25°), 1940 à aujourd'hui, monde. Variable utilisée :
  ``10m_wind_gust_since_previous_post_processing`` (rafale maximale de l'heure écoulée).
* ``sis-european-wind-storm-reanalysis`` : produit ``windstorm_track`` (trajectoires de tempêtes extratropicales, CSV, Europe/Atlantique Nord) ;
  paramètres vérifiés dans le catalogue : product, variable (all), tracking_algorithm (hodges, tempestextremes), year, month, day.

**Vent** (toutes rafales confondues) : maxima annuels de la rafale en un point → loi des valeurs extrêmes GEV (moments L, Hosking 1990) → rafale par
période de retour. **Tempête** : on retient, pour chaque tempête identifiée par les trajectoires, la rafale maximale observée au point dans une fenêtre
de ± 1 jour autour du pic ; les pics sont ajustés par une loi de Pareto généralisée (dépassements d'un seuil, moments L) avec le nombre moyen de
tempêtes par an → rafale de tempête par période de retour, et classement des tempêtes passées.

NON vérifié en réel depuis l'environnement de développement (réseau bloqué) : format exact des CSV renvoyés (les lecteurs sont tolérants et
signalent les colonnes trouvées), durée des files d'attente et limites de taille des requêtes (découpage automatique en périodes plus courtes).
Limites : résolution 0,25° (les rafales ERA5 sont lissées et sous-estiment souvent les extrêmes locaux) ; au-delà de la durée d'observation,
les périodes de retour longues sont des extrapolations ; la tempête n'est disponible que pour la zone 25°O–35°E, 30°N–70°N.
"""
from __future__ import annotations

import io
import math
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

CDS_URL = "https://cds.climate.copernicus.eu/api"
TS_DATASET = "reanalysis-era5-single-levels-timeseries"
ST_DATASET = "sis-european-wind-storm-reanalysis"
GUST_VAR = "10m_wind_gust_since_previous_post_processing"
GRID = 0.25
CACHE_DIR = Path(tempfile.gettempdir()) / "catmap_wind"
RPS = (2, 5, 10, 20, 50, 100, 200)
ALGORITHMS = ("tempestextremes", "hodges")
STORM_DOMAIN = (-25.0, 35.0, 30.0, 70.0)          # lon min, lon max, lat min, lat max (empreintes du jeu européen)
GUST_EDGES = [20.0, 28.0, 35.0]                   # classes de la carte (m/s)


class WindError(Exception):
    pass


def column_name(hazard: str, rp: int) -> str:
    """``AAA_BBB_CCC`` : aléa (vent | tempete), scénario, variable (ex. ``vent_RP100_gust``)."""
    return f"{hazard}_RP{int(rp)}_gust"


# ───────────────────────────── cellules de grille et client ─────────────────────────────
def cell_of(lat: float, lon: float) -> tuple[float, float]:
    """Point de grille ERA5 (0,25°) le plus proche ; longitude ramenée dans [-180, 180]."""
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    return (round(round(float(lat) / GRID) * GRID, 2), round(round(lon / GRID) * GRID, 2))


def make_client(key: str, url: str = CDS_URL):
    if not key:
        raise WindError("clé d'API Copernicus absente (secret `CDSAPI_KEY`).")
    try:
        import cdsapi
    except ImportError as e:
        raise WindError("bibliothèque « cdsapi » absente : ajoutez-la à requirements.txt.") from e
    return cdsapi.Client(url=url, key=key, quiet=True)


# ───────────────────────────── lecture des fichiers ─────────────────────────────
def _read_table(data: bytes) -> pd.DataFrame:
    if data[:2] == b"PK":                                        # archive zip : on concatène les CSV
        frames = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                if n.lower().endswith((".csv", ".txt")):
                    frames.append(_read_table(z.read(n)))
        if not frames:
            raise WindError("archive sans fichier CSV.")
        return pd.concat(frames, ignore_index=True)
    text = data.decode("utf-8-sig", "replace")
    try:
        df = pd.read_csv(io.StringIO(text), comment="#", sep=None, engine="python")
    except Exception as e:                                       # noqa: BLE001
        raise WindError(f"CSV illisible ({type(e).__name__}: {str(e)[:100]}).") from e
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_gust_series(data: bytes) -> pd.Series:
    """Série horaire de rafales (m/s), index datetime naïf (UTC)."""
    df = _read_table(data)
    low = {c: c.lower() for c in df.columns}
    tcol = next((c for c in df.columns if low[c] in ("valid_time", "time", "datetime", "date")), None) or \
        next((c for c in df.columns if "time" in low[c] or "date" in low[c]), None)
    vcol = next((c for c in df.columns if "fg10" in low[c] or "gust" in low[c]), None)
    if vcol is None:
        vcol = next((c for c in df.columns if c != tcol and not any(k in low[c] for k in ("lat", "lon")) and
                     pd.api.types.is_numeric_dtype(df[c])), None)
    if tcol is None or vcol is None:
        raise WindError("colonnes temps / rafale introuvables ; colonnes reçues : " + ", ".join(df.columns))
    t = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    v = pd.to_numeric(df[vcol], errors="coerce")
    s = pd.Series(v.to_numpy(), index=t).dropna()
    s = s[~s.index.isna()]
    if s.empty:
        raise WindError("aucune valeur de rafale exploitable dans le fichier.")
    s.index = s.index.tz_convert(None)
    return s.sort_index()


def daily_max(s: pd.Series) -> pd.Series:
    d = s.resample("D").max()
    return d.astype("float32")


def save_daily(path: Path, daily: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, start=np.array(str(daily.index[0].date())), values=daily.to_numpy().astype("float16"))


def load_daily(path: Path) -> pd.Series:
    z = np.load(path, allow_pickle=False)
    vals = z["values"].astype("float32")
    return pd.Series(vals, index=pd.date_range(str(z["start"]), periods=len(vals), freq="D"))


def annual_maxima(daily: pd.Series, min_days: int = 330) -> pd.Series:
    """Maximum annuel de la rafale, pour les années au moins presque complètes."""
    g = daily.groupby(daily.index.year)
    n, m = g.count(), g.max()
    return m[n >= min_days].dropna()


# ───────────────────────────── statistiques des extrêmes (moments L) ─────────────────────────────
def lmoments(x) -> tuple[float, float, float]:
    """Moments L l1, l2, l3 (Hosking 1990) d'un échantillon."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n < 4:
        raise WindError("échantillon trop court pour un ajustement.")
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    return b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0


def gev_fit(x) -> dict:
    """GEV par moments L (convention de Hosking : k > 0 = queue supérieure bornée)."""
    l1, l2, l3 = lmoments(x)
    if l2 <= 0:
        raise WindError("variabilité nulle : ajustement impossible.")
    t3 = l3 / l2
    c = 2.0 / (3.0 + t3) - math.log(2) / math.log(3)
    k = 7.8590 * c + 2.9554 * c * c
    k = float(np.clip(k, -0.5, 0.5))                              # évite les ajustements instables sur petits échantillons
    if abs(k) < 1e-6:
        alpha = l2 / math.log(2)
        xi = l1 - 0.5772156649 * alpha
    else:
        g = math.gamma(1 + k)
        alpha = l2 * k / ((1 - 2 ** (-k)) * g)
        xi = l1 + alpha * (g - 1) / k
    return {"xi": xi, "alpha": alpha, "k": k, "n": len(x)}


def gev_return_level(p: dict, T: float) -> float:
    F = 1.0 - 1.0 / T
    y = -math.log(F)
    if abs(p["k"]) < 1e-6:
        return p["xi"] - p["alpha"] * math.log(y)
    return p["xi"] + p["alpha"] / p["k"] * (1 - y ** p["k"])


def gpd_fit(exceedances) -> dict:
    """Pareto généralisée par moments L (Hosking : k = l1/l2 − 2, alpha = (1 + k) l1)."""
    y = np.asarray(exceedances, float)
    l1, l2, _ = lmoments(y)
    if l2 <= 0:
        raise WindError("dépassements sans variabilité.")
    k = float(np.clip(l1 / l2 - 2.0, -0.5, 0.5))
    return {"alpha": (1 + k) * l1, "k": k, "n": len(y)}


def gpd_return_level(p: dict, u: float, lam: float, T: float) -> float:
    """Niveau de retour T ans : seuil u, ``lam`` dépassements par an."""
    m = lam * T
    if m <= 1:
        return u
    if abs(p["k"]) < 1e-6:
        return u + p["alpha"] * math.log(m)
    return u + p["alpha"] / p["k"] * (1 - m ** (-p["k"]))


def return_levels_annual(annual: pd.Series, rps, min_years: int = 30) -> dict:
    """Rafale par période de retour d'après les maxima annuels (GEV). ``{rp: valeur | None}``."""
    if len(annual) < min_years:
        return {r: None for r in rps}
    p = gev_fit(annual.to_numpy())
    return {r: round(gev_return_level(p, r), 2) for r in rps}


def return_levels_storms(peaks, n_years: float, rps, min_events: int = 20, per_year: float = 1.0) -> dict:
    """Rafale de tempête par période de retour : seuil choisi pour environ ``per_year`` tempêtes dépassantes par an."""
    x = np.sort(np.asarray([v for v in peaks if v is not None and np.isfinite(v)], float))[::-1]
    if len(x) < min_events or n_years <= 0:
        return {r: None for r in rps}
    k = int(min(len(x) - 1, max(10, round(per_year * n_years))))
    u = float(x[k])
    exc = x[:k] - u
    exc = exc[exc > 0]
    if len(exc) < 8:
        return {r: None for r in rps}
    lam = len(exc) / n_years
    p = gpd_fit(exc)
    return {r: round(gpd_return_level(p, u, lam, r), 2) for r in rps}


# ───────────────────────────── requêtes CDS ─────────────────────────────
def build_timeseries_request(lat: float, lon: float, y0: int, y1: int) -> dict:
    return {"variable": [GUST_VAR], "location": {"longitude": float(lon), "latitude": float(lat)},
            "date": [f"{int(y0)}-01-01/{int(y1)}-12-31"], "data_format": "csv"}


def build_tracks_request(algorithm: str, y0: int, y1: int, variant: int = 0) -> dict:
    """Requête de trajectoires. Le CDS refuse (400 « valid combination ») les champs qu'un produit n'utilise pas ; comme le jeu de
    contraintes réel n'est pas lisible ici, on essaie des variantes de la plus simple à la plus complète (voir ``TRACK_VARIANTS``)."""
    req = {"product": "windstorm_track", "variable": "all", "tracking_algorithm": [algorithm],
           "year": [str(y) for y in range(int(y0), int(y1) + 1)]}
    if variant >= 1:
        req["month"] = [f"{m:02d}" for m in range(1, 13)]
    if variant >= 2:
        req["day"] = [f"{d:02d}" for d in range(1, 32)]
    if variant == 3:
        req["event_aggregation"] = "single_event"
    return req


TRACK_VARIANTS = (0, 1, 2, 3)


_TOO_BIG = re.compile(r"too large|cost|limit|exceed|too many|request is larger", re.I)


def _retrieve(client, dataset: str, request: dict, tmp: Path) -> bytes:
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = client.retrieve(dataset, request, str(tmp))
        path = Path(str(out)) if out is not None and Path(str(out)).exists() else tmp
        data = path.read_bytes()
    except Exception as e:                                       # noqa: BLE001
        raise WindError(f"{type(e).__name__}: {str(e)[:900]}") from e
    finally:
        tmp.unlink(missing_ok=True)
    if not data:
        raise WindError("réponse vide.")
    return data


def fetch_daily(client, lat: float, lon: float, y0: int, y1: int, cache_dir: Path = CACHE_DIR, min_span: int = 10) -> pd.Series:
    """Rafale maximale journalière au point de grille ; découpe la période en deux si la requête est refusée comme trop grande."""
    try:
        data = _retrieve(client, TS_DATASET, build_timeseries_request(lat, lon, y0, y1), cache_dir / f"tmp_{lat}_{lon}_{y0}_{y1}.dat")
        return daily_max(parse_gust_series(data))
    except WindError as e:
        if _TOO_BIG.search(str(e)) and (y1 - y0 + 1) > 2 * min_span:
            mid = (y0 + y1) // 2
            a = fetch_daily(client, lat, lon, y0, mid, cache_dir, min_span)
            b = fetch_daily(client, lat, lon, mid + 1, y1, cache_dir, min_span)
            return pd.concat([a, b]).sort_index().groupby(level=0).max()
        raise


def cell_path(cell: tuple[float, float], y0: int, y1: int, cache_dir: Path = CACHE_DIR) -> Path:
    return Path(cache_dir) / f"gust_{cell[0]:+.2f}_{cell[1]:+.2f}_{y0}_{y1}.npz"


def ensure_cells(cells, y0: int, y1: int, client, cache_dir: Path = CACHE_DIR, budget_s: float = 240.0,
                 on_progress: Optional[Callable[[int, int], None]] = None, workers: int = 4) -> dict:
    """Récupère (avec cache disque) les séries journalières des cellules, ``workers`` requêtes à la fois (le CDS traite plusieurs
    requêtes en parallèle). Aucun nouveau lot n'est lancé une fois le budget de temps consommé : relancer reprend là où on s'est arrêté.
    Renvoie ``{"done": [...], "pending": [...], "errors": {cellule: message}}``."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    done, pending, errors = [], [], {}
    cells = list(dict.fromkeys(cells))
    todo = []
    for cell in cells:
        (done if cell_path(cell, y0, y1, cache_dir).exists() else todo).append(cell)

    def one(cell):
        try:
            save_daily(cell_path(cell, y0, y1, cache_dir), fetch_daily(client, cell[0], cell[1], y0, y1, cache_dir))
            return cell, None
        except WindError as e:
            return cell, str(e)

    n_total, n_seen = len(cells), len(done)
    if on_progress:
        on_progress(n_seen, n_total)
    w = max(1, int(workers))
    for i in range(0, len(todo), w):
        batch = todo[i:i + w]
        if time.time() - t0 > budget_s:
            pending += batch
            continue
        with ThreadPoolExecutor(max_workers=w) as ex:
            for cell, err in ex.map(one, batch):
                if err is None:
                    done.append(cell)
                else:
                    errors[cell] = err
        n_seen += len(batch)
        if on_progress:
            on_progress(n_seen, n_total)
    return {"done": done, "pending": pending, "errors": errors}


def cache_info(cache_dir: Path = CACHE_DIR) -> tuple[int, float]:
    fs = list(Path(cache_dir).glob("*")) if Path(cache_dir).exists() else []
    return len(fs), round(sum(f.stat().st_size for f in fs if f.is_file()) / 1e6, 1)


def clear_cache(cache_dir: Path = CACHE_DIR) -> int:
    n = 0
    if Path(cache_dir).exists():
        for f in Path(cache_dir).glob("*"):
            if f.is_file():
                f.unlink(missing_ok=True)
                n += 1
    return n


# ───────────────────────────── tempêtes : trajectoires ─────────────────────────────
def parse_tracks(data: bytes) -> pd.DataFrame:
    """Trajectoires → colonnes ``storm, time, lon, lat, gust`` (gust peut être NaN si absente)."""
    df = _read_table(data)
    low = {c: c.lower() for c in df.columns}
    col = lambda *keys: next((c for c in df.columns if any(k in low[c] for k in keys)), None)  # noqa: E731
    sid = col("track_id", "storm_id", "storm", "event", "track") or next((c for c in df.columns if low[c] in ("id", "number")), None)
    tcol = col("valid_time", "datetime", "time", "date")
    lon, lat = col("lon"), col("lat")
    gust = col("gust", "fg10")
    if tcol is None and all(k in low.values() for k in ("year", "month", "day")):
        inv = {v: k for k, v in low.items()}
        parts = {k: df[inv[k]] for k in ("year", "month", "day")}
        parts["hour"] = df[inv["hour"]] if "hour" in inv else 0
        df["_t"] = pd.to_datetime(pd.DataFrame(parts), errors="coerce")
        tcol = "_t"
    if not (sid and tcol and lon and lat):
        raise WindError("colonnes des trajectoires non reconnues ; colonnes reçues : " + ", ".join(df.columns))
    out = pd.DataFrame({"storm": df[sid].astype(str), "time": pd.to_datetime(df[tcol], errors="coerce", utc=True),
                        "lon": pd.to_numeric(df[lon], errors="coerce"), "lat": pd.to_numeric(df[lat], errors="coerce"),
                        "gust": pd.to_numeric(df[gust], errors="coerce") if gust else np.nan})
    out["time"] = out["time"].dt.tz_convert(None)
    out["lon"] = ((out["lon"] + 180.0) % 360.0) - 180.0
    out = out.dropna(subset=["time", "lon", "lat"])
    if out.empty:
        raise WindError("aucune trajectoire exploitable.")
    return out


def fetch_tracks(client, algorithm: str, y0: int, y1: int, cache_dir: Path = CACHE_DIR, chunk: int = 20) -> pd.DataFrame:
    """Trajectoires (cache disque par période de ``chunk`` ans)."""
    frames = []
    for a in range(int(y0), int(y1) + 1, chunk):
        b = min(a + chunk - 1, int(y1))
        p = Path(cache_dir) / f"tracks_{algorithm}_{a}_{b}.csv"
        if p.exists() and p.stat().st_size:
            frames.append(parse_tracks(p.read_bytes()))
            continue
        data, errs = None, []
        for v in TRACK_VARIANTS:
            try:
                data = _retrieve(client, ST_DATASET, build_tracks_request(algorithm, a, b, v), Path(cache_dir) / f"tmp_tracks_{algorithm}_{a}_{b}.dat")
                break
            except WindError as e:
                errs.append(f"variante {v} : {e}")
                if not re.search(r"valid combination|invalid request|400|Bad Request", str(e), re.I):
                    raise
        if data is None:
            raise WindError("trajectoires refusées par Copernicus avec toutes les variantes de requête. " + " || ".join(errs)[:1800])
        tr = parse_tracks(data)
        p.parent.mkdir(parents=True, exist_ok=True)
        tr.to_csv(p, index=False)
        frames.append(tr)
    return pd.concat(frames, ignore_index=True)


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def extract_storms(tracks: pd.DataFrame, cells, radius_km: float = 1000.0, domain=STORM_DOMAIN) -> pd.DataFrame:
    """Tempêtes dont la trajectoire passe à moins de ``radius_km`` d'une cellule du portefeuille, dans le domaine des empreintes.
    Colonnes : ``storm, peak_time, peak_lon, peak_lat, track_gust, min_dist_km``. Le pic = point de la trajectoire à la rafale maximale
    (à défaut, le milieu de la trajectoire dans le domaine)."""
    lon0, lon1, lat0, lat1 = domain
    tr = tracks[(tracks["lon"].between(lon0, lon1)) & (tracks["lat"].between(lat0, lat1))]
    if tr.empty or not cells:
        return pd.DataFrame(columns=["storm", "peak_time", "peak_lon", "peak_lat", "track_gust", "min_dist_km"])
    clat = np.array([c[0] for c in cells])[None, :]
    clon = np.array([c[1] for c in cells])[None, :]
    dist = _haversine_km(tr["lat"].to_numpy()[:, None], tr["lon"].to_numpy()[:, None], clat, clon).min(axis=1)
    tr = tr.assign(dist=dist)
    rows = []
    for sid, g in tr.groupby("storm", sort=False):
        if g["dist"].min() > radius_km:
            continue
        g = g.sort_values("time")
        idx = g["gust"].idxmax() if g["gust"].notna().any() else g.index[len(g) // 2]
        pk = g.loc[idx]
        rows.append({"storm": sid, "peak_time": pk["time"], "peak_lon": float(pk["lon"]), "peak_lat": float(pk["lat"]),
                     "track_gust": float(pk["gust"]) if pd.notna(pk["gust"]) else np.nan, "min_dist_km": float(g["dist"].min())})
    cols = ["storm", "peak_time", "peak_lon", "peak_lat", "track_gust", "min_dist_km"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("peak_time").reset_index(drop=True)


def storm_gust(daily: pd.Series, peak_time, half_window_days: int = 1) -> float:
    """Rafale maximale au point dans la fenêtre ± ``half_window_days`` jours autour du pic de la tempête (NaN si pas de donnée)."""
    d = pd.Timestamp(peak_time).normalize()
    w = daily.reindex(pd.date_range(d - pd.Timedelta(days=half_window_days), d + pd.Timedelta(days=half_window_days), freq="D"))
    return float(w.max()) if w.notna().any() else float("nan")


def event_matrix(storms: pd.DataFrame, daily_by_cell: dict, point_cells: list) -> np.ndarray:
    """Matrice (tempêtes × points) des rafales de tempête."""
    cache = {c: np.array([storm_gust(daily_by_cell[c], t) if c in daily_by_cell else np.nan for t in storms["peak_time"]])
             for c in set(point_cells)}
    return np.column_stack([cache[c] for c in point_cells]) if len(point_cells) else np.empty((len(storms), 0))


def event_losses(gust: np.ndarray, classes, values, curves: dict) -> np.ndarray:
    """Perte par tempête (€) = somme sur les points de taux(rafale) × valeur exposée, avec la courbe de la classe d'actif du point."""
    classes = np.asarray(classes)
    vals = np.nan_to_num(np.asarray(values, float), nan=0.0)
    loss = np.zeros(gust.shape[0])
    for cls, curve in curves.items():
        idx = np.nonzero(classes == cls)[0]
        if len(idx):
            ratio = np.nan_to_num(curve.at(gust[:, idx]), nan=0.0)
            loss += (ratio * vals[idx][None, :]).sum(axis=1)
    return loss


# ───────────────────────────── repères pour la lecture ─────────────────────────────
# Dates du pic (jour UTC) de tempêtes européennes connues, pour nommer les événements du classement. Repères indicatifs : l'appariement se
# fait à ± 1 jour près et n'est pas fourni par le jeu Copernicus (qui ne donne que des identifiants de trajectoire).
KNOWN_STORMS = {"1990-01-25": "Daria", "1990-02-26": "Vivian", "1999-12-03": "Anatol", "1999-12-26": "Lothar", "1999-12-28": "Martin",
                "2007-01-18": "Kyrill", "2009-01-24": "Klaus", "2010-02-28": "Xynthia"}


def known_name(peak_time, tol_days: int = 1) -> str:
    d = pd.Timestamp(peak_time).normalize()
    for s, n in KNOWN_STORMS.items():
        if abs((d - pd.Timestamp(s)).days) <= tol_days:
            return n
    return ""


GUST_COLORS = {0: "#fff3b0", 1: "#fdae61", 2: "#f46d43", 3: "#a50026", None: "#9e9e9e"}
GUST_LABELS = {0: "< 20 m/s", 1: "20 – 28 m/s", 2: "28 – 35 m/s", 3: "≥ 35 m/s"}


def gust_class(v):
    if v is None or not np.isfinite(v):
        return None
    return int(sum(v >= e for e in GUST_EDGES))


def cells_of_points(lat, lon) -> list:
    """Cellule de grille par point (``None`` si coordonnées invalides)."""
    out = []
    num = lambda x: pd.to_numeric(pd.Series(x).astype(str).str.replace(",", ".", regex=False), errors="coerce")  # noqa: E731  (virgule décimale)
    for a, b in zip(num(lat), num(lon)):
        out.append(cell_of(a, b) if np.isfinite(a) and np.isfinite(b) and abs(a) <= 90 and abs(b) <= 360 else None)
    return out


def compute_vent(daily_by_cell: dict, rps, min_years: int = 30) -> dict:
    """``{cellule: {rp: rafale | None}}`` par GEV sur les maxima annuels."""
    out = {}
    for c, d in daily_by_cell.items():
        try:
            out[c] = return_levels_annual(annual_maxima(d), rps, min_years)
        except WindError:
            out[c] = {r: None for r in rps}
    return out


def compute_tempete(storms: pd.DataFrame, daily_by_cell: dict, n_years: float, rps, min_events: int = 20) -> tuple[dict, dict]:
    """Retourne (``{cellule: {rp: rafale | None}}``, ``{cellule: rafales de chaque tempête}``)."""
    levels, peaks = {}, {}
    for c, d in daily_by_cell.items():
        pk = np.array([storm_gust(d, t) for t in storms["peak_time"]]) if len(storms) else np.array([])
        peaks[c] = pk
        try:
            levels[c] = return_levels_storms(pk[np.isfinite(pk)], n_years, rps, min_events)
        except WindError:
            levels[c] = {r: None for r in rps}
    return levels, peaks
