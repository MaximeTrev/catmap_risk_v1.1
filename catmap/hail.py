"""Aléa grêle par point : grêle très grosse (≥ 5 cm), fréquence annuelle d'événements par maille ERA5 de 0,25°.

Source : Battaglioli, Taszarek, Groenemeijer, Púčik & Rädler (2025), *Contrasting trends in very large hail events and related economic losses across
the globe*, Nature Geoscience (doi:10.1038/s41561-025-01868-0). Données : Zenodo, enregistrement 17379738 (ESSL, licence CC BY-NC-ND 4.0 : usage non
commercial, aucune redistribution de version modifiée ; les fichiers sont lus tels quels et jamais republiés).
Modèle de l'article : régression logistique (AR-CHaMo) appliquée à la réanalyse ERA5 (prédicteurs ThundeR), probabilité d'éclair × probabilité
conditionnelle de grêle ≥ 5 cm, toutes les 3 h, monde entier à 0,25°, 1950–2023.

Deux approches (au choix de l'utilisateur) :

* ``clim`` : climatologie 1950–2023, nombre moyen annuel d'événements ≥ 5 cm par maille (``hail_ge5cm_climatology_1950_2023.nc``). Pas de période de retour ;
  la perte est une perte annuelle moyenne.
* ``annual`` : 74 fichiers annuels ``prob_hail_ge5cm_<année>.nc``. On lit la valeur de chaque année à la maille du point, puis on en déduit le niveau
  annuel dépassé une année sur ``RP`` par quantile empirique (positions de Weibull, sans extrapolation ; le nombre d'années limite la période de retour
  maximale). Les valeurs annuelles sont stockées par maille dans un petit cache : le gros fichier est supprimé après lecture.

Option « tendance » : le fichier ``hail_ge5cm_trends_1950_2023.nc`` donne la tendance et sa p-value par maille. La fréquence est multipliée par un facteur
≥ 1 (tendance à la hausse) ou ≤ 1 (à la baisse) si la tendance est significative (seuils réglables).

NON vérifié depuis l'environnement de développement (réseau bloqué) : noms de variables des fichiers annuels, unité de la tendance (par an ou par
décennie) et signification exacte (nombre attendu d'événements par an) des valeurs annuelles. ``diagnose`` affiche ces éléments sur des villes connues.
Dommage : taux par événement × valeur exposée, taux ILLUSTRATIF à calibrer (aucune courbe de vulnérabilité grêle mondiale libre n'est citée ici).
"""
from __future__ import annotations

import json
import math
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import damage as dmg
from .damage import Curve

RISK = "grele"
ZENODO_RECORD = "17379738"
CACHE_DIR = Path(tempfile.gettempdir()) / "catmap_grele"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "grele"
COMPACT_NAME = "hail_compact.npz"
CLIM_FILE = "hail_ge5cm_climatology_1950_2023.nc"
TREND_FILE = "hail_ge5cm_trends_1950_2023.nc"
YEARS = tuple(range(1950, 2024))
GRID = 0.25
RPS = (2, 5, 10, 20, 50)
MODES = {"clim": "Climatologie (fréquence moyenne, sans période de retour)",
         "annual": "Séries annuelles 1950–2023 (périodes de retour)"}
FREQ_EDGES = [0.05, 0.2, 0.5]                                  # événements ≥ 5 cm par an et par maille : classes de la carte (indicatives)
FREQ_COLORS = {0: "#fff3b0", 1: "#fdae61", 2: "#f46d43", 3: "#a50026", None: "#9e9e9e"}
FREQ_LABELS = {0: "< 0,05 / an", 1: "0,05 – 0,2 / an", 2: "0,2 – 0,5 / an", 3: "≥ 0,5 / an"}
HDF5_MAGIC, CDF_MAGIC = b"\x89HDF", b"CDF"
CHECKS = {"Mendoza (Argentine)": (-32.9, -68.8), "Dallas": (32.8, -96.8), "Milan": (45.5, 9.2), "Paris": (48.85, 2.35),
          "Brisbane": (-27.5, 153.0), "Johannesbourg": (-26.2, 28.05)}


class HailError(Exception):
    pass


def annual_file(year: int) -> str:
    return f"prob_hail_ge5cm_{int(year)}.nc"


def column_name(rp: int) -> str:
    """``aléa_scénario_variable`` : ``grele_RP50_freq`` (événements ≥ 5 cm par an et par maille)."""
    return f"{RISK}_RP{int(rp)}_freq"


CLIM_COL = f"{RISK}_moy_freq"


def freq_class(v):
    if v is None or not np.isfinite(v):
        return None
    return int(sum(v >= e for e in FREQ_EDGES))


# ───────────────────────────── lecture NetCDF ─────────────────────────────
def _fill(arr: np.ndarray, attrs: dict) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind not in "fiu":
        return a
    a = a.astype(float)
    for k in ("_FillValue", "missing_value"):
        v = attrs.get(k)
        if v is not None:
            for fv in np.atleast_1d(v):
                try:
                    a = np.where(a == float(fv), np.nan, a)
                except (TypeError, ValueError):
                    pass
    sc, off = attrs.get("scale_factor"), attrs.get("add_offset")
    if sc is not None:
        a = a * float(np.atleast_1d(sc)[0])
    if off is not None:
        a = a + float(np.atleast_1d(off)[0])
    return a


def _txt(v):
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def read_dataset(path) -> dict:
    """Lit un fichier NetCDF (NetCDF4/HDF5 ou NetCDF3) : ``{"attrs", "vars": {nom: {"dims", "data", "attrs"}}}``. Essaie netCDF4, h5netcdf, scipy."""
    path = str(path)
    errors = []
    try:
        import netCDF4
        ds = netCDF4.Dataset(path)
        try:
            vs = {}
            for n, v in ds.variables.items():
                at = {k: v.getncattr(k) for k in v.ncattrs()}
                vs[n] = {"dims": tuple(v.dimensions), "data": np.ma.filled(np.ma.asarray(v[:]).astype(float), np.nan)
                         if v.dtype.kind in "fiu" else np.asarray(v[:]), "attrs": at}
            return {"attrs": {k: ds.getncattr(k) for k in ds.ncattrs()}, "vars": vs}
        finally:
            ds.close()
    except ImportError:
        errors.append("netCDF4 absent")
    except Exception as e:                                      # noqa: BLE001
        errors.append(f"netCDF4 : {type(e).__name__} {str(e)[:80]}")
    try:
        import h5netcdf
        with h5netcdf.File(path, "r") as ds:
            vs = {}
            for n, v in ds.variables.items():
                at = {k: v.attrs[k] for k in v.attrs}
                vs[n] = {"dims": tuple(v.dimensions), "data": _fill(v[...], at), "attrs": at}
            return {"attrs": {k: ds.attrs[k] for k in ds.attrs}, "vars": vs}
    except ImportError:
        errors.append("h5netcdf absent")
    except Exception as e:                                      # noqa: BLE001
        errors.append(f"h5netcdf : {type(e).__name__} {str(e)[:80]}")
    try:
        from scipy.io import netcdf_file
        with netcdf_file(path, "r", mmap=False) as ds:
            vs = {}
            for n, v in ds.variables.items():
                at = {k: _txt(getattr(v, k)) for k in getattr(v, "_attributes", {})}
                vs[n] = {"dims": tuple(v.dimensions), "data": _fill(np.array(v[:]), at), "attrs": at}
            return {"attrs": {k: _txt(getattr(ds, k)) for k in getattr(ds, "_attributes", {})}, "vars": vs}
    except Exception as e:                                      # noqa: BLE001
        errors.append(f"scipy : {type(e).__name__} {str(e)[:80]}")
    raise HailError("lecture NetCDF impossible (" + " ; ".join(errors) + "). Ajoutez « h5netcdf » et « h5py » (ou « netCDF4 ») à requirements.txt.")


def _axis(ds: dict, names: tuple) -> Optional[tuple]:
    for n in names:
        for k, v in ds["vars"].items():
            if k.lower() == n and v["data"].ndim == 1:
                return k, np.asarray(v["data"], float)
    return None


def field(ds: dict, prefer: tuple = (), contains: tuple = ()) -> dict:
    """Champ 2-D (lat, lon) d'un jeu lu : ``{"name", "lat", "lon", "z", "attrs"}``. Variable choisie par nom préféré, sinon par mot contenu, sinon la seule candidate."""
    la, lo = _axis(ds, ("lat", "latitude", "y")), _axis(ds, ("lon", "longitude", "x"))
    if la is None or lo is None:
        raise HailError("axes lat/lon introuvables ; variables : " + ", ".join(ds["vars"]))
    cands = {k: v for k, v in ds["vars"].items() if v["data"].ndim >= 2 and la[0] in v["dims"] and lo[0] in v["dims"]}
    if not cands:
        raise HailError("aucune variable (lat, lon) ; variables : " + ", ".join(f"{k}{v['dims']}" for k, v in ds["vars"].items()))
    name = next((p for p in prefer if p in cands), None) or next((k for k in cands if any(c in k.lower() for c in contains)), None) or next(iter(cands))
    v = cands[name]
    z = np.asarray(v["data"], float)
    dims = list(v["dims"])
    keep = [i for i, d in enumerate(dims) if d in (la[0], lo[0])]
    for i in sorted((i for i in range(len(dims)) if i not in keep), reverse=True):
        if z.shape[i] != 1:
            raise HailError(f"variable {name} : dimension supplémentaire {dims[i]} de taille {z.shape[i]} (attendu : 1) ; dimensions {tuple(dims)}")
        z = np.take(z, 0, axis=i)
        dims.pop(i)
    if dims.index(la[0]) > dims.index(lo[0]):
        z = z.T
    return {"name": name, "lat": la[1], "lon": lo[1], "z": z, "attrs": v["attrs"]}


def _step(a: np.ndarray) -> float:
    d = np.abs(np.diff(np.sort(a)))
    return float(np.median(d)) if len(d) else GRID


def sample(fld: dict, lat, lon) -> np.ndarray:
    """Valeur de la maille la plus proche ; NaN hors de l'emprise (au-delà d'une demi-maille) ou si la maille est sans donnée."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    la, lo = fld["lat"], fld["lon"]
    lon_adj = np.where((lo.max() > 180) & (lon < 0), lon + 360.0, lon)
    i = np.abs(la[None, :] - lat[:, None]).argmin(axis=1) if len(lat) else np.array([], int)
    j = np.abs(lo[None, :] - lon_adj[:, None]).argmin(axis=1) if len(lat) else np.array([], int)
    out = fld["z"][i, j].astype(float) if len(lat) else np.array([], float)
    tol_la, tol_lo = 0.51 * _step(la), 0.51 * _step(lo)
    far = (np.abs(la[i] - lat) > tol_la) | (np.abs(lo[j] - lon_adj) > tol_lo)
    return np.where(far | ~np.isfinite(out), np.nan, out)


# ───────────────────────────── téléchargement Zenodo ─────────────────────────────
def _zenodo_files(session=None, record: str = ZENODO_RECORD) -> dict:
    """``{nom: adresse}`` d'après l'API publique de Zenodo ; vide si indisponible."""
    try:
        r = (session or __import__("requests")).get(f"https://zenodo.org/api/records/{record}", timeout=60, headers={"User-Agent": "CatMap"})
        if r.status_code != 200:
            return {}
        out = {}
        for f in (r.json().get("files") or []):
            links = f.get("links") or {}
            url, name = links.get("self") or links.get("download"), f.get("key") or f.get("filename") or ""
            if url and name:
                out[name] = url
        return out
    except Exception:                                           # noqa: BLE001
        return {}


def _local(name: str, cache_dir: Path, data_dir: Path) -> Optional[Path]:
    for d in (Path(data_dir), Path(cache_dir)):
        p = d / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _try_download(u: str, name: str, cache_dir: Path, session) -> tuple:
    """Télécharge une adresse ; renvoie (chemin, None) ou (None, message d'erreur)."""
    tmp = Path(cache_dir) / f"{name}.{threading.get_ident()}.part"
    try:
        r = (session or __import__("requests")).get(u, stream=True, timeout=(15, 120), headers={"User-Agent": "CatMap"})
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        head = b""
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if not head:
                    head = chunk[:200]
                f.write(chunk)
    except Exception as e:                                       # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return None, f"{type(e).__name__} {str(e)[:100]}"
    if head[:4] == HDF5_MAGIC or head[:3] == CDF_MAGIC:
        dest = Path(cache_dir) / name
        tmp.replace(dest)
        return dest, None
    tmp.unlink(missing_ok=True)
    return None, f"contenu ni NetCDF ni HDF5 (début : {head[:60]!r})"


def fetch_file(name: str, session=None, cache_dir: Path = CACHE_DIR, data_dir: Path = DATA_DIR, urls: Optional[dict] = None) -> Path:
    """Fichier du jeu : data/grele/, sinon cache, sinon téléchargement Zenodo (contenu vérifié : NetCDF/HDF5). Adresse directe d'abord (rapide) ;
    l'API Zenodo (liste des fichiers) n'est interrogée qu'en cas d'échec."""
    p = _local(name, cache_dir, data_dir)
    if p:
        return p
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    path, err = _try_download(f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1", name, cache_dir, session)
    if path:
        return path
    errors.append(err)
    urls = urls if urls is not None else _zenodo_files(session)
    if name in urls:
        path, err = _try_download(urls[name], name, cache_dir, session)
        if path:
            return path
        errors.append(err)
    raise HailError(f"téléchargement de {name} impossible : " + " | ".join(errors)[:600] +
                    f" — vous pouvez déposer le fichier dans data/grele/ (Zenodo {ZENODO_RECORD}).")


# ───────────────────────────── fichier compact (précalculé) ─────────────────────────────
# Les 74 fichiers annuels pèsent ≈ 480 Mo : les télécharger un à un est long. ``build_compact`` (outil ``tools/build_hail_compact.py``, à lancer une fois
# sur un poste qui dispose des fichiers) en tire un seul fichier ``data/grele/hail_compact.npz`` : climatologie (moyenne annuelle), tendance, p-value
# et valeurs annuelles des mailles où la grêle n'est pas négligeable. Quand il est présent, plus aucun téléchargement n'est nécessaire.
_COMPACT_CACHE: dict = {}


def compact_path(data_dir: Path = DATA_DIR) -> Optional[Path]:
    p = Path(data_dir) / COMPACT_NAME
    return p if p.is_file() and p.stat().st_size > 0 else None


def _compact(path: Path) -> dict:
    key = (str(path), path.stat().st_mtime)
    if key not in _COMPACT_CACHE:
        try:
            with np.load(path) as z:
                d = {k: z[k] for k in z.files}
        except Exception as e:                                   # noqa: BLE001
            raise HailError(f"fichier compact illisible ({type(e).__name__}: {str(e)[:100]}) : {path.name}") from e
        miss = [k for k in ("lat", "lon", "years", "clim", "trend", "pvalue", "cells", "annual", "scale") if k not in d]
        if miss:
            raise HailError(f"fichier compact incomplet (manque : {', '.join(miss)}).")
        _COMPACT_CACHE.clear()
        _COMPACT_CACHE[key] = d
    return _COMPACT_CACHE[key]


def compact_years(path: Path) -> list:
    return [int(y) for y in _compact(path)["years"]]


def compact_fields(path: Path) -> dict:
    d = _compact(path)
    return {k: {"name": k, "lat": d["lat"].astype(float), "lon": d["lon"].astype(float), "z": d[k].astype(float), "attrs": {}}
            for k in ("clim", "trend", "pvalue")}


def compact_series(path: Path, cells: list) -> dict:
    """``{clé de maille: valeurs annuelles (dans l'ordre de compact_years)}``. Maille absente du stock mais valide (grêle négligeable) : zéros ;
    maille sans donnée : NaN."""
    d = _compact(path)
    la, lo, kept = d["lat"].astype(float), d["lon"].astype(float), d["cells"]
    ann, scale = d["annual"], float(d["scale"])
    out = {}
    for c in cells:
        i, j = int(np.abs(la - c[0]).argmin()), int(np.abs(lo - c[1]).argmin())
        key = _cell_key(c)
        if abs(la[i] - c[0]) > 0.13 or abs(lo[j] - c[1]) > 0.13 or not np.isfinite(d["clim"][i, j]):
            out[key] = np.full(ann.shape[0], np.nan)
            continue
        pos = int(np.searchsorted(kept, i * len(lo) + j))
        out[key] = ann[:, pos].astype(float) / scale if pos < len(kept) and kept[pos] == i * len(lo) + j else np.zeros(ann.shape[0])
    return out


def build_compact(src_dir, out_path, min_clim: float = 0.0005, max_mb: float = 90.0, progress=print) -> dict:
    """Construit le fichier compact à partir des NetCDF d'origine déposés dans ``src_dir`` (climatologie, tendances, fichiers annuels).
    Valeurs annuelles stockées en entiers 16 bits (précision 1/scale). Mailles conservées : moyenne annuelle ≥ ``min_clim`` événements/an ; le seuil est
    relevé automatiquement jusqu'à ce que le fichier tienne sous ``max_mb`` Mo."""
    src = Path(src_dir)
    years = [y for y in YEARS if (src / annual_file(y)).is_file()]
    if not years:
        raise HailError(f"aucun fichier {annual_file(2000)} (etc.) dans {src}")
    progress(f"{len(years)} fichier(s) annuel(s) trouvé(s)")
    arrs, ref = [], None
    for y in years:
        f = field(read_dataset(src / annual_file(y)), contains=("hail", "prob"))
        ref = ref or f
        if f["z"].shape != ref["z"].shape:
            raise HailError(f"grille différente en {y}")
        arrs.append(f["z"].astype(np.float32))
        if len(arrs) % 10 == 0:
            progress(f"  {len(arrs)}/{len(years)} lus")
    A = np.stack(arrs)
    mean = np.nanmean(A, axis=0)
    tds = read_dataset(src / TREND_FILE)
    tr = field(tds, prefer=("trend_hail_ge5cm",), contains=("trend",))["z"]
    pv = field(tds, prefer=("pvalue_hail_ge5cm",), contains=("pvalue", "p_value"))["z"]
    cfile = field(read_dataset(src / CLIM_FILE), prefer=("hail_ge5cm",), contains=("hail",))["z"] if (src / CLIM_FILE).is_file() else None
    if cfile is not None:
        ok = np.isfinite(cfile) & (mean > 0)
        progress(f"rapport climatologie du fichier ÷ moyenne des années : {float(np.median(cfile[ok] / mean[ok])):.1f} (≈ {len(years)} si la climatologie est un cumul)")
    top = float(np.nanmax(A))
    scale = float(10 ** math.floor(math.log10(65000.0 / top))) if top > 0 else 1.0
    nlon = len(ref["lon"])
    thr, info = float(min_clim), {}
    for _ in range(6):
        keep = np.isfinite(mean) & (mean >= thr)
        flat = np.flatnonzero(keep.ravel()).astype(np.int32)
        q = np.nan_to_num(A.reshape(len(years), -1)[:, flat] * scale, nan=0.0)
        q = np.clip(np.rint(q), 0, 65535).astype(np.uint16)
        mask = keep
        np.savez_compressed(out_path, lat=ref["lat"].astype(np.float32), lon=ref["lon"].astype(np.float32), years=np.array(years, np.int16),
                            clim=mean.astype(np.float32), trend=np.where(mask, tr, np.nan).astype(np.float32),
                            pvalue=np.where(mask, pv, np.nan).astype(np.float32), cells=flat, annual=q, scale=np.float32(scale))
        mb = Path(out_path).stat().st_size / 1e6
        info = {"annees": len(years), "mailles_conservees": int(len(flat)), "mailles_total": int(np.isfinite(mean).sum()), "taille_mo": round(mb, 1),
                "seuil_moyenne_annuelle": thr, "precision": 1.0 / scale}
        progress(f"seuil {thr:g} événements/an : {len(flat)} mailles, {mb:.1f} Mo")
        if mb <= max_mb:
            return info
        thr *= 3
    raise HailError(f"fichier compact trop gros ({info.get('taille_mo')} Mo) même avec un seuil élevé.")


# ───────────────────────────── points et mailles ─────────────────────────────
def cell_of(lat: float, lon: float) -> tuple:
    la = round(float(lat) * 4) / 4
    lo = round(float(lon) * 4) / 4
    if lo >= 180:
        lo -= 360.0
    return (la, lo)


def _cell_key(c: tuple) -> str:
    return f"{c[0]:.2f},{c[1]:.2f}"


def _points(df: pd.DataFrame, lat_col: str, lon_col: str):
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy(float)  # noqa: E731
    lat, lon = num(lat_col), num(lon_col)
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    return lat, lon, valid


# ───────────────────────────── séries annuelles (cache par année) ─────────────────────────────
def _year_path(cache_dir: Path, year: int) -> Path:
    return Path(cache_dir) / "series" / f"{int(year)}.json"


def _load_year(cache_dir: Path, year: int) -> dict:
    p = _year_path(cache_dir, year)
    if not p.exists():
        return {}
    try:
        return {k: (np.nan if v is None else float(v)) for k, v in json.loads(p.read_text()).items()}
    except ValueError:
        return {}


def _save_year(cache_dir: Path, year: int, d: dict) -> None:
    p = _year_path(cache_dir, year)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: (None if not np.isfinite(v) else round(float(v), 6)) for k, v in d.items()}))


def ensure_years(cells: list, years=YEARS, session=None, cache_dir: Path = CACHE_DIR, data_dir: Path = DATA_DIR, budget_s: float = 200.0,
                 on_progress=None, workers: int = 4) -> dict:
    """Garantit, pour chaque année, la valeur de chaque maille demandée. Ordre : cache de séries ; fichier compact (data/grele/, instantané) ; sinon
    téléchargement du fichier de l'année (``workers`` en parallèle), lecture des seules mailles demandées puis suppression du gros fichier (sauf s'il
    vient de data/grele/). Temps limité : reprise au prochain appel. Renvoie ``{"done", "pending", "errors", "source"}``."""
    keys = [_cell_key(c) for c in cells]
    t0 = time.time()
    done, errors, source = 0, {}, "cache"
    todo = []
    for y in years:
        have = _load_year(cache_dir, y)
        if all(k in have for k in keys):
            done += 1
        else:
            todo.append(y)
    cp = compact_path(data_dir) if todo else None
    if cp:
        try:
            cy = compact_years(cp)
            ser = compact_series(cp, cells)
            left = []
            for y in todo:
                if y in cy:
                    have = _load_year(cache_dir, y)
                    have.update({k: float(ser[k][cy.index(y)]) for k in keys})
                    _save_year(cache_dir, y, have)
                    done += 1
                else:
                    left.append(y)
            todo, source = left, "compact"
        except HailError as e:
            errors["compact"] = str(e)[:300]
    if todo:
        source = "zenodo" if source == "cache" else source + "+zenodo"
        stop = threading.Event()
        la, lo = [c[0] for c in cells], [c[1] for c in cells]

        def work(y):
            if stop.is_set() or time.time() - t0 > budget_s:
                return y, None
            path = None
            try:
                path = fetch_file(annual_file(y), session, cache_dir, data_dir)
                vals = sample(field(read_dataset(path), contains=("hail", "prob")), la, lo)
                have = _load_year(cache_dir, y)
                have.update(dict(zip(keys, vals)))
                _save_year(cache_dir, y, have)
                return y, None
            except HailError as e:
                return y, str(e)[:400]
            finally:
                if path is not None and Path(path).parent == Path(cache_dir):
                    Path(path).unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            n = 0
            for fut in as_completed([ex.submit(work, y) for y in todo]):
                y, err = fut.result()
                n += 1
                if err:
                    errors[str(y)] = err
                    if len(errors) >= 3:                       # même cause probable : inutile d'insister
                        stop.set()
                if on_progress:
                    on_progress(done + n, len(years))
        done = sum(all(k in _load_year(cache_dir, y) for k in keys) for y in years)
    return {"done": done, "pending": len(years) - done, "errors": errors, "source": source}


def load_series(cells: list, years=YEARS, cache_dir: Path = CACHE_DIR) -> dict:
    """``{clé de maille: tableau des valeurs annuelles}`` pour les années en cache."""
    data = {y: _load_year(cache_dir, y) for y in years}
    return {_cell_key(c): np.array([data[y].get(_cell_key(c), np.nan) for y in years], float) for c in cells}


def series_cache_cells(cache_dir: Path = CACHE_DIR, years=YEARS) -> list:
    """Clés des mailles présentes dans toutes les années en cache."""
    sets = [set(_load_year(cache_dir, y)) for y in years]
    sets = [s for s in sets if s]
    return sorted(set.intersection(*sets)) if sets else []


def rp_levels(series, rps=RPS) -> dict:
    """Niveau annuel dépassé en moyenne une année sur ``RP`` : quantile empirique, positions de Weibull ``i/(n+1)``, sans extrapolation
    (``NaN`` si ``RP > n + 1``)."""
    x = np.sort(np.asarray(series, float)[np.isfinite(series)])
    n = len(x)
    out = {}
    for rp in rps:
        if n < 3 or rp > n + 1:
            out[rp] = np.nan
            continue
        pos = np.arange(1, n + 1) / (n + 1.0)
        out[rp] = float(np.interp(1.0 - 1.0 / rp, pos, x))
    return out


def empirical_curve(series) -> pd.DataFrame:
    """Points (période de retour empirique, valeur) : ``RP = (n+1)/rang`` en classant les années de la plus forte à la plus faible."""
    x = np.sort(np.asarray(series, float)[np.isfinite(series)])[::-1]
    n = len(x)
    return pd.DataFrame({"rp": (n + 1.0) / np.arange(1, n + 1), "valeur": x})


# ───────────────────────────── tendance ─────────────────────────────
def trend_factor(freq, trend, pvalue, per_decade: bool = True, years: float = 38.0, p_max: float = 0.05, rel_min_pct: float = 0.0,
                 fmin: float = 0.5, fmax: float = 2.0) -> np.ndarray:
    """Facteur multiplicatif de la fréquence : ``(f + pente × années) / f`` borné à [fmin, fmax], appliqué seulement si la tendance est significative
    (``p ≤ p_max``) et si sa valeur relative par décennie atteint ``rel_min_pct`` %. Sinon 1. ``years`` : écart entre le milieu de la période (1986,5)
    et l'année visée."""
    f, tr, p = (np.asarray(a, float) for a in (freq, trend, pvalue))
    slope = tr / (10.0 if per_decade else 1.0)                      # par an
    with np.errstate(divide="ignore", invalid="ignore"):
        rel10 = np.where(f > 0, slope * 10.0 / f * 100.0, np.nan)    # % par décennie
        raw = np.where(f > 0, (f + slope * years) / f, 1.0)
    ok = np.isfinite(f) & (f > 0) & np.isfinite(tr) & np.isfinite(p) & (p <= p_max) & (np.abs(np.nan_to_num(rel10)) >= rel_min_pct)
    fac = np.where(ok, np.clip(raw, fmin, fmax), 1.0)
    return np.where(np.isfinite(f), fac, np.nan)


def threshold_table(freq, trend, pvalue, per_decade: bool = True, years: float = 38.0, fmin: float = 0.5, fmax: float = 2.0) -> pd.DataFrame:
    """Aide au choix du seuil : pour chaque couple (p-value max, tendance relative minimale), nombre de points ajustés (hausse / baisse) et facteur moyen."""
    rows = []
    for p_max in (0.01, 0.05, 0.10):
        for rel in (0, 5, 10, 20):
            fac = trend_factor(freq, trend, pvalue, per_decade, years, p_max, rel, fmin, fmax)
            ok = np.isfinite(fac)
            rows.append({"p-value max": p_max, "tendance relative min (%/décennie)": rel, "points ajustés à la hausse": int((fac[ok] > 1).sum()),
                         "points ajustés à la baisse": int((fac[ok] < 1).sum()), "facteur moyen": round(float(np.mean(fac[ok])), 3) if ok.any() else None})
    return pd.DataFrame(rows)


# ───────────────────────────── estimation de l'aléa ─────────────────────────────
def estimate(df: pd.DataFrame, lat_col: str, lon_col: str, mode: str = "clim", rps=RPS, use_trend: bool = False, trend_opts: Optional[dict] = None,
             session=None, cache_dir: Path = CACHE_DIR, data_dir: Path = DATA_DIR, budget_s: float = 200.0,
             clim_total: bool = True) -> tuple[pd.DataFrame, dict]:
    """Ajoute la fréquence de grêle ≥ 5 cm (événements par an et par maille). Mode ``clim`` : ``grele_moy_freq`` ; mode ``annual`` : ``grele_RP<rp>_freq``.
    Avec la tendance : la colonne utilisée pour les dommages est ajustée, la valeur d'origine est conservée dans ``…_freq_brut``."""
    if mode not in MODES:
        raise HailError(f"mode inconnu : {mode}")
    out = df.copy().reset_index(drop=True)
    lat, lon, valid = _points(out, lat_col, lon_col)
    idx = np.nonzero(valid)[0]
    cells = sorted({cell_of(lat[i], lon[i]) for i in idx})
    summ = {"points": len(out), "invalides": int((~valid).sum()), "mode": mode, "rps": [], "en_attente": 0, "erreurs": {}, "cellules": len(cells),
            "tendance": bool(use_trend)}
    status = ["coordonnées invalides"] * len(out)
    base: dict = {}
    clim_vals = None
    cp = compact_path(data_dir)
    cfields = compact_fields(cp) if cp else None
    summ["source"] = "fichier compact" if cp else "Zenodo"
    if mode == "clim" or use_trend:
        if cfields:                                          # moyenne annuelle déjà calculée à partir des 74 fichiers annuels
            cf, div = cfields["clim"], 1.0
        else:
            cf = field(read_dataset(fetch_file(CLIM_FILE, session, cache_dir, data_dir)), prefer=("hail_ge5cm",), contains=("hail",))
            div = float(len(YEARS)) if clim_total else 1.0     # le fichier de climatologie est un cumul 1950–2023 (rapport ≈ 64 constaté vs année 2000)
        clim_vals = np.full(len(out), np.nan)
        clim_vals[idx] = sample(cf, lat[idx], lon[idx]) / div
    if mode == "clim":
        base[CLIM_COL] = clim_vals
        summ["colonnes"] = [CLIM_COL]
    else:
        rps = [r for r in rps if r <= len(YEARS) + 1]
        if len(rps) == 0:
            raise HailError("aucune période de retour exploitable (maximum : 75 ans avec 74 années).")
        res = ensure_years(cells, YEARS, session, cache_dir, data_dir, budget_s)
        summ.update({"en_attente": res["pending"], "erreurs": res["errors"], "annees_ok": res["done"], "rps": list(rps), "source": res["source"]})
        for r in rps:
            base[column_name(r)] = np.full(len(out), np.nan)
        if res["pending"] == 0:
            ser = load_series(cells, YEARS, cache_dir)
            lv = {k: rp_levels(v, rps) for k, v in ser.items()}
            for i in idx:
                k = _cell_key(cell_of(lat[i], lon[i]))
                for r in rps:
                    base[column_name(r)][i] = lv[k][r]
        summ["colonnes"] = [column_name(r) for r in rps]
    if use_trend:
        if cfields:
            tr, pv = cfields["trend"], cfields["pvalue"]
        else:
            tf = read_dataset(fetch_file(TREND_FILE, session, cache_dir, data_dir))
            tr = field(tf, prefer=("trend_hail_ge5cm",), contains=("trend",))
            pv = field(tf, prefer=("pvalue_hail_ge5cm",), contains=("pvalue", "p_value"))
        tval, pval = np.full(len(out), np.nan), np.full(len(out), np.nan)
        tval[idx], pval[idx] = sample(tr, lat[idx], lon[idx]), sample(pv, lat[idx], lon[idx])
        o = dict(trend_opts or {})
        o.setdefault("per_decade", True)
        fac = trend_factor(clim_vals, tval, pval, **o)
        out[f"{RISK}_tendance"], out[f"{RISK}_tendance_p"], out[f"{RISK}_facteur_tendance"] = tval, pval, np.round(fac, 4)
        if mode == "annual":
            out[CLIM_COL] = np.round(clim_vals, 5)               # fréquence moyenne 1950–2023, référence de la tendance
        for c in list(base):
            out[c + "_brut"] = base[c]
            base[c] = base[c] * np.where(np.isfinite(fac), fac, 1.0)
        summ["facteur_moyen"] = float(np.nanmean(fac)) if np.isfinite(fac).any() else None
        summ["hausse"], summ["baisse"] = int(np.nansum(fac > 1)), int(np.nansum(fac < 1))
    for c, v in base.items():
        out[c] = np.round(v, 5)
    first = base[summ["colonnes"][0]]
    pend = summ["en_attente"] > 0
    for i in idx:
        status[i] = ("en attente (années à télécharger)" if pend else
                     "ok" if np.isfinite(first[i]) else "sans donnée (mer, zone non couverte ou hors de la grille)")
    out[f"{RISK}_statut"] = status
    ok = np.array([s == "ok" for s in status])
    summ["ok"] = int(ok.sum())
    summ["sans_donnee"] = int(len(out) - ok.sum() - (~valid).sum()) if not pend else 0
    summ["freq_max"] = float(np.nanmax(first)) if np.isfinite(first).any() else None
    return out, summ


# ───────────────────────────── dommages ─────────────────────────────
def event_curve(ratio_per_event: float, freq_max: float = 20.0) -> Curve:
    """Taux de dommage = ``min(1, taux par événement × fréquence)`` (courbe linéaire plafonnée à 1) ; abscisse : événements ≥ 5 cm par an."""
    if not 0 < ratio_per_event <= 1:
        raise ValueError("le taux de dommage par événement doit être compris entre 0 et 1.")
    top = max(1.0 / ratio_per_event, 1.0)
    return Curve([0.0, top], [0.0, 1.0], f"Linéaire : {ratio_per_event:.2%} de la valeur par événement ≥ 5 cm (taux illustratif), plafonné à 100 %")


def estimate_damages(base: pd.DataFrame, summ: dict, ratio_per_event: float, classes: pd.Series, values: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Mode annuel : ``dommage_grele_RP<rp>_eur`` et PAM (trapèzes entre les périodes de retour). Mode climatologie : une seule perte, qui est la perte
    annuelle moyenne ``dommage_grele_PAM_eur`` (= min(1, taux × fréquence moyenne) × valeur)."""
    if ratio_per_event is None or not 0 < ratio_per_event <= 1:
        raise dmg.DamageError("le taux de dommage par événement doit être compris entre 0 et 1.")
    label = f"Grêle ≥ 5 cm : {ratio_per_event:.2%} par événement (illustratif)"
    if summ["mode"] == "annual":
        curve = event_curve(ratio_per_event)
        cols = {r: column_name(r) for r in summ["rps"] if column_name(r) in base.columns}
        if not cols:
            raise dmg.DamageError("aucune colonne de fréquence disponible : estimez d'abord l'aléa.")
        return dmg.estimate_damages(base, cols, {c: curve for c in classes.dropna().unique()}, classes, values, label, None, RISK)
    freq = pd.to_numeric(base[CLIM_COL], errors="coerce")
    ratio = (freq * ratio_per_event).clip(upper=1.0)
    o = pd.DataFrame(index=base.index)
    o["dommage_methode"], o["dommage_classe"], o["dommage_valeur_exposee_eur"] = label, classes, values
    o[f"dommage_{RISK}_moy_ratio"] = ratio
    o[dmg.column_pam(RISK)] = ratio * values
    summ_d = {"points": len(base), "sans_valeur": int(values.isna().sum()), "sans_hauteur": int(freq.isna().sum()), "classes_sans_courbe": [],
              "pertes_totales": {}, "pam_totale": float(np.nansum(o[dmg.column_pam(RISK)])), "pam_bornes": None, "repli_global": 0}
    return o, summ_d


# ───────────────────────────── diagnostic ─────────────────────────────
def _describe(f: dict, label: str, extra_attrs=True) -> dict:
    la, lo = f["lat"], f["lon"]
    z = f["z"]
    at = " ; ".join(f"{k} = {str(v)[:120]}" for k, v in f["attrs"].items()) if extra_attrs else ""
    return {"étape": label, "résultat": "OK",
            "détail": f"variable « {f['name']} », grille {len(la)}×{len(lo)}, lat {la.min():g} à {la.max():g} (pas {_step(la):g}), lon {lo.min():g} à {lo.max():g} "
                      f"(pas {_step(lo):g}), valeurs {np.nanmin(z):.3g} à {np.nanmax(z):.3g}, moyenne {np.nanmean(z):.3g}" + (f". Attributs : {at}" if at else "")}


def diagnose(session=None, cache_dir: Path = CACHE_DIR, data_dir: Path = DATA_DIR) -> list[dict]:
    """Télécharge (si besoin) la climatologie, la tendance et UNE année, et décrit leur contenu : sert à confirmer noms de variables, grille, unités."""
    rows = []
    cp = compact_path(data_dir)
    if cp:
        try:
            yrs = compact_years(cp)
            d = _compact(cp)
            rows.append({"étape": "fichier compact", "résultat": "OK", "détail": f"{cp.name} ({cp.stat().st_size / 1e6:.1f} Mo) : {len(yrs)} années "
                         f"({yrs[0]}–{yrs[-1]}), {len(d['cells'])} mailles conservées, précision {1 / float(d['scale']):g}. "
                         "Plus aucun téléchargement n'est nécessaire."})
        except HailError as e:
            rows.append({"étape": "fichier compact", "résultat": "ÉCHEC", "détail": str(e)})
    files = {}
    fl = {}
    for label, name, prefer, cont in (("climatologie", CLIM_FILE, ("hail_ge5cm",), ("hail",)), ("année 2000", annual_file(2000), (), ("hail", "prob"))):
        try:
            p = fetch_file(name, session, cache_dir, data_dir, files)
            ds = read_dataset(p)
            f = field(ds, prefer=prefer, contains=cont)
            fl[label] = f
            rows.append(_describe(f, f"fichier {label} ({name}, {p.stat().st_size / 1e6:.1f} Mo)"))
            if ds["attrs"]:
                rows.append({"étape": f"attributs globaux ({label})", "résultat": "OK", "détail": " ; ".join(f"{k} = {str(v)[:200]}" for k, v in ds["attrs"].items())})
        except HailError as e:
            rows.append({"étape": f"fichier {label}", "résultat": "ÉCHEC", "détail": str(e)})
    try:
        tds = read_dataset(fetch_file(TREND_FILE, session, cache_dir, data_dir, files))
        for nm, pref, cont in (("tendance", ("trend_hail_ge5cm",), ("trend",)), ("p-value", ("pvalue_hail_ge5cm",), ("pvalue", "p_value"))):
            f = field(tds, prefer=pref, contains=cont)
            fl[nm] = f
            rows.append(_describe(f, f"fichier tendances : {nm}"))
        rows.append({"étape": "unité de la tendance", "résultat": "À CONTRÔLER",
                     "détail": "lisez les attributs ci-dessus (« units » ou « Description ») : par an ou par décennie ? Réglez l'unité dans l'application en conséquence."})
    except HailError as e:
        rows.append({"étape": "fichier tendances", "résultat": "ÉCHEC", "détail": str(e)})
    if fl:
        pts = list(CHECKS.values())
        la, lo = [p[0] for p in pts], [p[1] for p in pts]
        got = {k: sample(f, la, lo) for k, f in fl.items()}
        for i, nm in enumerate(CHECKS):
            rows.append({"étape": f"valeurs à {nm}", "résultat": "OK" if any(np.isfinite(v[i]) for v in got.values()) else "À CONTRÔLER",
                         "détail": " ; ".join(f"{k} : {'aucune valeur' if not np.isfinite(v[i]) else format(v[i], '.4g')}" for k, v in got.items())})
        if "climatologie" in got and "année 2000" in got:
            c, a = got["climatologie"], got["année 2000"]
            ok = np.isfinite(c) & np.isfinite(a) & (c > 0)
            r = float(np.mean(c[ok] / a[ok])) if ok.any() else float("nan")
            if not ok.any():
                verdict, txt = "À CONTRÔLER", "non calculable"
            elif 0.2 < r < 5:
                verdict, txt = "OK", f"climatologie ÷ année 2000 = {r:.1f} : la climatologie est une moyenne annuelle."
            elif 20 < r < 200:
                verdict, txt = "OK", (f"climatologie ÷ année 2000 = {r:.1f} (≈ {len(YEARS)} ans) : la climatologie du fichier est un CUMUL sur la période, "
                                      f"pas une moyenne annuelle ; l'application la divise par {len(YEARS)} (option modifiable).")
            else:
                verdict, txt = "À CONTRÔLER", f"climatologie ÷ année 2000 = {r:.1f} : rapport inattendu, vérifier la signification des fichiers."
            rows.append({"étape": "cohérence climatologie / année 2000", "résultat": verdict, "détail": txt})
    return rows
