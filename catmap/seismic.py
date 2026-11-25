"""Aléa sismique par point : accélération du sol (PGA, en g) par période de retour.

Jeux de données :

* **GEM Global Seismic Hazard Map v2023.1** (Zenodo 8409647, licence CC BY-NC-SA 4.0 : usage non commercial). Monde, un seul niveau : PGA
  à 10 % de probabilité de dépassement en 50 ans (période de retour 475 ans), sol rocheux (Vs30 760–800 m/s), raster ``v2023_1_pga_475_rock_3min.tif``
  (3 minutes d'arc ≈ 5 km). Le zip (≈ 35 Mo) est téléchargé à la demande et le raster extrait dans le cache ; un fichier déjà placé dans
  ``data/seisme/`` est utilisé en priorité (le raster de 173 Mo dépasse la limite des dépôts GitHub classiques).
* **ESHM20 (Europe, EFEHR, CC BY 4.0)** : service web ``efehr-services.ethz.ch/hazard/api`` ; courbe d'aléa PGA par point (``/v1/curves``), d'où la PGA
  pour toute période de retour comprise dans la courbe (interpolation log-log, sans extrapolation). Points regroupés par maille de 0,05°.

NON vérifié depuis l'environnement de développement (réseau bloqué) : l'adresse de téléchargement du zip, l'organisation interne du TIFF (codec,
tuiles) et l'unité exacte des valeurs (attendue : fraction de g). ``diagnose`` contrôle ces points sur des villes connues.

Dommage : fonction de vulnérabilité lognormale ``taux = Φ(ln(PGA/θ)/β)`` (``vulnerability_curve``). Les paramètres par défaut sont ILLUSTRATIFS et
doivent être calibrés par l'utilisateur selon le type de construction : ils ne proviennent d'aucune publication.
"""
from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import floodhazard as fhz
from .damage import Curve

RISK = "seisme"
CACHE_DIR = Path(tempfile.gettempdir()) / "catmap_seisme"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "seisme"
GEM_ZIP_URL = "https://zenodo.org/records/8409647/files/GEM-GSHM_PGA-475y-rock_v2023.zip?download=1"
GEM_TIF = "v2023_1_pga_475_rock_3min.tif"
GEM_RPS = (475,)
ESHM_URL = "https://efehr-services.ethz.ch/hazard/api"
ESHM_OPENAPI = ESHM_URL + "/openapi.json"
ESHM_RPS = (50, 100, 200, 475, 975, 2475)
DATASETS = {"gem": {"label": "GEM – carte mondiale (475 ans)", "rps": GEM_RPS},
            "eshm20": {"label": "ESHM20 – Europe (plusieurs périodes de retour)", "rps": ESHM_RPS}}
PGA_EDGES = [0.08, 0.16, 0.32]                                  # g : classes de la carte
PGA_COLORS = {0: "#fff3b0", 1: "#fdae61", 2: "#f46d43", 3: "#a50026", None: "#9e9e9e"}
PGA_LABELS = {0: "< 0,08 g", 1: "0,08 – 0,16 g", 2: "0,16 – 0,32 g", 3: "≥ 0,32 g"}


class SeismicError(Exception):
    pass


def column_name(rp: int) -> str:
    """``aléa_scénario_variable`` : ``seisme_RP475_pga``."""
    return f"{RISK}_RP{int(rp)}_pga"


def pga_class(v):
    if v is None or not np.isfinite(v):
        return None
    return int(sum(v >= e for e in PGA_EDGES))


# ───────────────────────────── fichier GEM ─────────────────────────────
ZENODO_RECORD = "8409647"
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def _zenodo_files(session=None) -> list[tuple[str, str, int]]:
    """Fichiers du dépôt Zenodo (nom, adresse, taille) d'après l'API publique ; liste vide si indisponible."""
    try:
        r = (session or __import__("requests")).get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60, headers={"User-Agent": "CatMap"})
        if r.status_code != 200:
            return []
        out = []
        for f in (r.json().get("files") or []):
            links = f.get("links") or {}
            url = links.get("self") or links.get("download")
            name = f.get("key") or f.get("filename") or ""
            if url and name:
                out.append((name, url, int(f.get("size") or 0)))
        return out
    except Exception:                                           # noqa: BLE001
        return []


def _download_gem(cache_dir: Path, session, url: str) -> Path:
    """Télécharge le jeu GEM : essaie les fichiers listés par l'API Zenodo (zip de préférence, plus léger, sinon le .tif), puis l'adresse devinée.
    Vérifie le contenu réel (zip ou TIFF) et non le seul code HTTP ; renvoie le chemin du zip ou du .tif obtenu."""
    files = _zenodo_files(session)
    zips = [f for f in files if f[0].lower().endswith(".zip")]
    tifs = [f for f in files if f[0].lower().endswith((".tif", ".tiff")) and "pga" in f[0].lower()]
    cands = [(n, u) for n, u, _ in zips + tifs] + [("gem.zip", url)]
    errors = []
    for name, u in dict.fromkeys(cands):
        tmp = Path(cache_dir) / (Path(name).name + ".part")
        try:
            r = (session or __import__("requests")).get(u, stream=True, timeout=180, headers={"User-Agent": "CatMap"})
            if r.status_code != 200:
                errors.append(f"{name} : HTTP {r.status_code}")
                continue
            head = b""
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if not head:
                        head = chunk[:200]
                    f.write(chunk)
        except Exception as e:                                   # noqa: BLE001
            errors.append(f"{name} : {type(e).__name__} {str(e)[:100]}")
            tmp.unlink(missing_ok=True)
            continue
        if head[:4] in TIFF_MAGIC:
            dest = Path(cache_dir) / (Path(name).name if name.lower().endswith((".tif", ".tiff")) else GEM_TIF)
            tmp.replace(dest)
            return dest
        if zipfile.is_zipfile(tmp):
            dest = Path(cache_dir) / "gem_pga475.zip"
            tmp.replace(dest)
            return dest
        errors.append(f"{name} : contenu ni zip ni TIFF (début : {head[:80]!r})")
        tmp.unlink(missing_ok=True)
    raise SeismicError("téléchargement GEM impossible. " + " | ".join(errors)[:900] +
                       " — vous pouvez déposer le zip (35 Mo) dans data/seisme/ sur GitHub.")


def gem_path(cache_dir: Path = CACHE_DIR, session=None, data_dir: Path = DATA_DIR, url: str = GEM_ZIP_URL) -> Path:
    """Chemin du raster GEM : data/seisme/, sinon cache, sinon téléchargement du zip Zenodo puis extraction."""
    for d in (Path(data_dir), Path(cache_dir)):
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in (".tif", ".tiff") and "pga" in p.name.lower() and p.stat().st_size > 0:
                    return p
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zpath = cache_dir / "gem_pga475.zip"
    local_zip = next((p for p in sorted(Path(data_dir).glob("*.zip")) if "pga" in p.name.lower() or "gem" in p.name.lower()), None) \
        if Path(data_dir).exists() else None
    if local_zip is not None and zipfile.is_zipfile(local_zip):        # zip déposé dans data/seisme/ (35 Mo : accepté par git, pas par l'envoi web)
        zpath = local_zip
    elif not (zpath.exists() and zipfile.is_zipfile(zpath)):
        got = _download_gem(cache_dir, session, url)
        if got.suffix.lower() in (".tif", ".tiff"):                      # le dépôt fournit directement le raster
            return got
        zpath = got
    try:
        with zipfile.ZipFile(zpath) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")) and "pga" in n.lower()] or \
                    [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not names:
                raise SeismicError("aucun fichier .tif dans l'archive GEM ; contenu : " + ", ".join(z.namelist())[:300])
            dest = cache_dir / Path(names[0]).name
            with z.open(names[0]) as src, open(dest, "wb") as out:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
    except zipfile.BadZipFile as e:
        zpath.unlink(missing_ok=True)
        raise SeismicError("archive GEM corrompue : relancez le calcul pour la retélécharger.") from e
    return dest


def _extent(path: Path) -> Optional[tuple]:
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            x0, y0, sx, sy = fhz._georef(page)
            return (x0, y0 - sy * int(page.imagelength), x0 + sx * int(page.imagewidth), y0)
    except Exception:                                          # noqa: BLE001
        return None


# ───────────────────────────── fichier précalculé GEM ─────────────────────────────
# Le raster GEM pèse 35 Mo (zip) / 173 Mo (tif) : le télécharger à chaque session est long. ``build_compact`` (outil ``tools/build_gem_compact.py``, à lancer
# une fois) le convertit en un petit fichier ``gem_pga475_compact.npz`` (PGA en entiers 16 bits, précision 1e-4 g) à déposer dans data/seisme/.
COMPACT_NAME = "gem_pga475_compact.npz"
_NODATA_CODE = 65535
_COMPACT_CACHE: dict = {}


def compact_path(data_dir: Path = DATA_DIR) -> Optional[Path]:
    for d in (Path(data_dir), Path(data_dir).parent):          # data/seisme/ de préférence, sinon directement data/
        p = d / COMPACT_NAME
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _compact(path: Path) -> dict:
    key = (str(path), path.stat().st_mtime)
    if key not in _COMPACT_CACHE:
        try:
            with np.load(path) as z:
                d = {k: z[k] for k in z.files}
        except Exception as e:                                   # noqa: BLE001
            raise SeismicError(f"fichier précalculé illisible ({type(e).__name__}: {str(e)[:100]}) : {Path(path).name}") from e
        miss = [k for k in ("pga", "origin", "pixel", "scale") if k not in d]
        if miss:
            raise SeismicError(f"fichier précalculé incomplet (manque : {', '.join(miss)}) : relancez tools/build_gem_compact.py.")
        _COMPACT_CACHE.clear()
        _COMPACT_CACHE[key] = d
    return _COMPACT_CACHE[key]


def compact_extent(path: Path) -> tuple:
    d = _compact(path)
    x0, y0 = (float(v) for v in d["origin"])
    sx, sy = (float(v) for v in d["pixel"])
    H, W = d["pga"].shape
    return (x0, y0 - sy * H, x0 + sx * W, y0)


def sample_compact(path: Path, lons, lats) -> list:
    """PGA (g) du pixel de chaque point ; ``None`` hors image ou sans donnée."""
    d = _compact(path)
    x0, y0 = (float(v) for v in d["origin"])
    sx, sy = (float(v) for v in d["pixel"])
    a, scale = d["pga"], float(d["scale"])
    H, W = a.shape
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    cols = np.floor((lons - x0) / sx).astype(np.int64)
    rows = np.floor((y0 - lats) / sy).astype(np.int64)
    out = []
    for r, c in zip(rows, cols):
        if 0 <= r < H and 0 <= c < W and a[r, c] != _NODATA_CODE:
            out.append(float(a[r, c]) / scale)
        else:
            out.append(None)
    return out


def read_full(tif: Path) -> tuple:
    """Lit tout le raster : ``(tableau float32 (lignes, colonnes) avec NaN pour sans-donnée, (x0, y0), (sx, sy))``."""
    import tifffile
    with tifffile.TiffFile(str(tif)) as tf:
        page = tf.pages[0]
        H, W = int(page.imagelength), int(page.imagewidth)
        x0, y0, sx, sy = fhz._georef(page)
        nd = fhz._nodata(page)
        try:
            arr = np.squeeze(page.asarray())
            if arr.ndim == 1 and arr.size == H * W:
                arr = arr.reshape(H, W)
        except Exception:                                       # noqa: BLE001 - repli : lecture bande par bande
            arr = None
        if arr is None or arr.shape != (H, W):
            if page.is_tiled:
                raise SeismicError("lecture du raster impossible (organisation en tuiles non lisible par tifffile).")
            rps_ = int(page.rowsperstrip)
            arr = np.full((H, W), np.nan, np.float32)
            fh = tf.filehandle
            for seg in range(len(page.dataoffsets)):
                if not page.databytecounts[seg]:
                    continue
                fh.seek(page.dataoffsets[seg])
                data, _idx, _ = page.decode(fh.read(page.databytecounts[seg]), seg)
                blk = np.squeeze(data)
                blk = blk.reshape(-1, W) if blk.ndim == 1 else blk
                arr[seg * rps_: seg * rps_ + blk.shape[0]] = blk[: H - seg * rps_]
        arr = arr.astype(np.float32)
        if nd is not None:
            arr[arr == nd] = np.nan
        arr[~np.isfinite(arr) | (arr < 0)] = np.nan
    return arr, (x0, y0), (sx, sy)


def build_compact(tif: Path, out_path: Path, max_mb: float = 90.0, progress=print) -> dict:
    """Convertit le raster GEM en fichier précalculé (PGA × ``scale`` en entiers 16 bits, 65535 = sans donnée)."""
    arr, origin, pixel = read_full(Path(tif))
    top = float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0
    if top <= 0:
        raise SeismicError("raster vide : aucune valeur de PGA positive.")
    scale = float(10 ** math.floor(math.log10(65000.0 / top)))
    q = np.where(np.isfinite(arr), np.clip(np.rint(arr * scale), 0, 65534), _NODATA_CODE).astype(np.uint16)
    np.savez_compressed(out_path, pga=q, origin=np.array(origin, np.float64), pixel=np.array(pixel, np.float64), scale=np.float32(scale))
    mb = Path(out_path).stat().st_size / 1e6
    progress(f"écrit : {out_path} ({mb:.1f} Mo) ; {q.shape[1]}×{q.shape[0]} pixels, PGA max {top:.3f} g, précision {1 / scale:g} g")
    if mb > max_mb:
        raise SeismicError(f"fichier précalculé trop gros ({mb:.0f} Mo > {max_mb:g} Mo).")
    return {"pixels": [int(q.shape[1]), int(q.shape[0])], "taille_mo": round(mb, 2), "pga_max": round(top, 4), "precision_g": 1.0 / scale}


def estimate_pga(df: pd.DataFrame, lat_col: str, lon_col: str, dataset: str = "gem", rps=GEM_RPS, session=None,
                 cache_dir: Path = CACHE_DIR, tif: Optional[Path] = None, budget_s: float = 200.0, use_compact: bool = True,
                 data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict]:
    """Ajoute ``seisme_RP<rp>_pga`` (g) et ``seisme_statut``. Les périodes de retour non disponibles pour le jeu sont ignorées."""
    if dataset == "eshm20":
        return _estimate_eshm(df, lat_col, lon_col, rps, session, cache_dir, budget_s)
    rps = [r for r in rps if r in GEM_RPS]
    if not rps:
        raise SeismicError("aucune période de retour disponible pour ce jeu (GEM : 475 ans uniquement).")
    out = df.copy().reset_index(drop=True)
    num = lambda c: pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy(float)  # noqa: E731
    lat, lon = num(lat_col), num(lon_col)
    valid = np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90) & (abs(lon) <= 180)
    idx = np.nonzero(valid)[0]
    cp = compact_path(data_dir) if (use_compact and not tif) else None
    if cp:                                                      # fichier précalculé : aucun téléchargement
        path, vals, ext = cp, sample_compact(cp, lon[idx], lat[idx]), compact_extent(cp)
    else:
        path = Path(tif) if tif else gem_path(cache_dir, session, data_dir)
        try:
            vals = fhz.sample_geotiff(path, lon[idx], lat[idx])
        except fhz.TileError as e:
            raise SeismicError(str(e)) from e
        ext = _extent(path)
    col = [np.nan] * len(out)
    status = ["coordonnées invalides"] * len(out)
    for i, v in zip(idx, vals):
        if ext is not None and not (ext[0] <= lon[i] <= ext[2] and ext[1] <= lat[i] <= ext[3]):
            status[i] = "hors du raster (hors de l'emprise de la carte)"
        elif v is None or not math.isfinite(v) or v < 0:
            status[i] = "sans donnée (mer ou zone non couverte)"
        else:
            col[i] = round(float(v), 4)
            status[i] = "ok"
    out[column_name(475)] = col
    out[f"{RISK}_statut"] = status
    ok = np.array([s == "ok" for s in status])
    summary = {"points": len(out), "invalides": int((~valid).sum()), "ok": int(ok.sum()), "sans_donnee": int(len(out) - ok.sum() - (~valid).sum()),
               "pga_max": float(np.nanmax(col)) if ok.any() else None, "fichier": str(path), "rps": rps, "en_attente": 0,
               "source": "fichier précalculé" if cp else "raster GEM brut"}
    return out, summary


# ───────────────────────────── ESHM20 : service web EFEHR ─────────────────────────────
# Routes (description lue dans le service) : GET /v1/models?lon&lat (modèles disponibles au point), GET /v1/curves?lon&lat&modelid&imt
# (courbe d'aléa du point), GET /v1/maps?lon&lat&modelid&imt&poe&timespan. On utilise les courbes : elles donnent n'importe quelle période
# de retour par interpolation. NON vérifié : forme exacte des réponses JSON ; ``parse_curve`` est tolérant et ``diagnose_eshm`` montre la réponse brute.
def _get_json(path: str, params: dict, session=None, timeout: int = 40):
    try:
        r = (session or __import__("requests")).get(ESHM_URL + path, params=params, timeout=timeout,
                                                      headers={"User-Agent": "CatMap", "Accept": "application/json"})
    except Exception as e:                                      # noqa: BLE001
        raise SeismicError(f"service EFEHR inaccessible ({type(e).__name__}: {str(e)[:150]})") from e
    if r.status_code != 200:
        raise SeismicError(f"EFEHR {path} : HTTP {r.status_code} {str(getattr(r, 'text', ''))[:200]}")
    try:
        return r.json()
    except ValueError as e:
        raise SeismicError(f"EFEHR {path} : réponse non JSON ({str(getattr(r, 'text', ''))[:120]})") from e


def _records(js) -> list:
    if isinstance(js, list):
        return js
    if isinstance(js, dict):
        for k in ("models", "data", "results", "items"):
            if isinstance(js.get(k), list):
                return js[k]
    return []


def pick_eshm_model(js) -> Optional[tuple[int, str]]:
    """(identifiant, nom) du modèle ESHM20 dans la réponse de ``/v1/models`` ; ``None`` s'il n'y figure pas."""
    for m in _records(js):
        if not isinstance(m, dict):
            continue
        text = " ".join(str(v) for v in m.values()).lower()
        mid = next((m[k] for k in ("id", "model_id", "modelid", "modelId") if k in m), None)
        if mid is not None and ("eshm20" in text or "eshm 20" in text or "eshm2020" in text):
            try:
                return int(mid), str(m.get("name") or m.get("title") or m.get("label") or "ESHM20")
            except (TypeError, ValueError):
                continue
    return None


def _num_list(v) -> Optional[list]:
    if isinstance(v, list) and len(v) >= 5 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return [float(x) for x in v]
    return None


def _collect(js, found: list, depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(js, dict):
        for k, v in js.items():
            nl = _num_list(v)
            if nl is not None:
                found.append((str(k), nl))
            else:
                _collect(v, found, depth + 1)
    elif isinstance(js, list):
        if js and all(isinstance(x, dict) for x in js):                      # liste d'enregistrements : une colonne par champ numérique
            keys = [k for k in js[0] if all(isinstance(x.get(k), (int, float)) and not isinstance(x.get(k), bool) for x in js)]
            if len(js) >= 5 and len(keys) >= 2:
                for k in keys:
                    found.append((str(k), [float(x[k]) for x in js]))
                return
        if js and all(isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(y, (int, float)) for y in x) for x in js) and len(js) >= 5:
            found.append(("x", [float(x[0]) for x in js]))
            found.append(("y", [float(x[1]) for x in js]))
            return
        for x in js:
            _collect(x, found, depth + 1)


def annual_from_window(poe, years: float):
    """Probabilité annuelle équivalente à une probabilité de dépassement ``poe`` sur ``years`` ans : ``1 − (1 − poe)^(1/years)``."""
    poe = np.clip(np.asarray(poe, float), 0.0, 1.0)
    return 1.0 - (1.0 - poe) ** (1.0 / float(years))


def pick_curve(curves: list) -> dict:
    """Parmi les courbes renvoyées par ``/v1/curves`` (plusieurs agrégations du modèle), retient la moyenne arithmétique si elle est proposée,
    sinon le quantile 0,5, sinon la première."""
    kind = lambda c: str(c.get("aggregation_type") or "").lower()          # noqa: E731
    for c in curves:
        if "arith" in kind(c) or "mean" in kind(c):
            return c
    for c in curves:
        if c.get("aggregation_level") == 0.5:
            return c
    return curves[0]


def parse_curve(js) -> tuple[np.ndarray, np.ndarray]:
    """Courbe d'aléa ``(niveaux de PGA croissants, probabilités annuelles décroissantes)``.

    Format réel du service EFEHR (observé) : ``{"curves": [{"investigation_time": 50, "aggregation_type": …, "points": [{"iml", "poe"}, …]}, …]}`` ;
    ``poe`` est la probabilité de dépassement sur ``investigation_time`` ans, convertie ici en probabilité annuelle. Pour toute autre forme,
    on cherche deux listes numériques de même longueur (niveaux croissants, probabilités décroissantes), supposées annuelles."""
    if isinstance(js, dict) and isinstance(js.get("curves"), list) and js["curves"] and isinstance(js["curves"][0], dict) \
            and isinstance(js["curves"][0].get("points"), list):
        c = pick_curve(js["curves"])
        pts = [q for q in c["points"] if isinstance(q, dict) and "iml" in q and "poe" in q]
        if len(pts) >= 2:
            iml = np.array([float(q["iml"]) for q in pts])
            poe = np.array([float(q["poe"]) for q in pts])
            order = np.argsort(iml)
            iml, poe = iml[order], poe[order]
            T = c.get("investigation_time")
            return iml, (annual_from_window(poe, T) if T else poe)
    found: list = []
    _collect(js, found)
    for i, (kx, x) in enumerate(found):
        if not all(b >= a for a, b in zip(x, x[1:])) or x[0] < 0:
            continue
        for ky, y in found:
            if len(y) == len(x) and y is not x and all(b <= a for a, b in zip(y, y[1:])) and min(y) >= 0 and max(y) <= 1.0 + 1e-9 and y[0] > y[-1]:
                return np.array(x, float), np.array(y, float)
    raise SeismicError("courbe d'aléa non reconnue dans la réponse du service (listes numériques trouvées : "
                       + ", ".join(f"{k}[{len(v)}]" for k, v in found[:8]) + ").")


def pga_at_rp(iml: np.ndarray, poe: np.ndarray, rp: float) -> Optional[float]:
    """PGA dont la probabilité annuelle de dépassement est ``1 − exp(−1/RP)`` (interpolation log-log, sans extrapolation)."""
    target = 1.0 - math.exp(-1.0 / float(rp))
    m = (iml > 0) & (poe > 0)
    x, y = np.log(iml[m]), np.log(poe[m])
    if len(x) < 2 or not (y.min() <= math.log(target) <= y.max()):
        return None
    order = np.argsort(y)
    return float(math.exp(np.interp(math.log(target), y[order], x[order])))


def _eshm_point(lon: float, lat: float, model_id: int, session=None, imt: str = "PGA"):
    js = _get_json("/v1/curves", {"lon": round(lon, 4), "lat": round(lat, 4), "modelid": model_id, "imt": imt}, session)
    return parse_curve(js)


def _estimate_eshm(df, lat_col, lon_col, rps, session, cache_dir, budget_s):
    import json as _json
    import time
    rps = [r for r in rps if r > 0]
    if not rps:
        raise SeismicError("choisissez au moins une période de retour.")
    out = df.copy().reset_index(drop=True)
    num = lambda c: pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy(float)  # noqa: E731
    lat, lon = num(lat_col), num(lon_col)
    valid = np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90) & (abs(lon) <= 180)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = lambda la, lo: (round(round(la / 0.05) * 0.05, 2), round(round(lo / 0.05) * 0.05, 2))   # maille de 0,05° (≈ 5 km) : une requête par maille
    cells = list(dict.fromkeys(key(a, b) for a, b, v in zip(lat, lon, valid) if v))
    cfile = cache_dir / "eshm20_curves.json"
    store = {}
    if cfile.exists():
        try:
            store = _json.loads(cfile.read_text())
        except ValueError:
            store = {}
    model_id = store.get("_model")
    t0 = time.time()
    errors: dict = {}
    pending = 0
    for c in cells:
        k = f"{c[0]},{c[1]}"
        if k in store:
            continue
        if time.time() - t0 > budget_s:
            pending += 1
            continue
        try:
            if model_id is None:
                found = pick_eshm_model(_get_json("/v1/models", {"lon": c[1], "lat": c[0]}, session))
                if found is None:
                    store[k] = {"err": "hors du domaine du modèle ESHM20 (Europe)"}
                    continue
                model_id = store["_model"] = found[0]
            iml, poe = _eshm_point(c[1], c[0], model_id, session)
            store[k] = {"iml": iml.tolist(), "poe": poe.tolist()}
        except SeismicError as e:
            msg = str(e)
            if "inaccessible" in msg or "HTTP 5" in msg or "HTTP 429" in msg:
                errors[k] = msg
                pending += 1
            else:
                store[k] = {"err": msg[:160]}
    cfile.write_text(_json.dumps(store))
    cols = {r: [np.nan] * len(out) for r in rps}
    status = ["coordonnées invalides"] * len(out)
    for i in range(len(out)):
        if not valid[i]:
            continue
        c = key(lat[i], lon[i])
        e = store.get(f"{c[0]},{c[1]}")
        if e is None:
            status[i] = "en attente (relancez le calcul)"
        elif "err" in e:
            status[i] = e["err"]
        else:
            iml, poe = np.array(e["iml"]), np.array(e["poe"])
            got = 0
            for r in rps:
                v = pga_at_rp(iml, poe, r)
                if v is not None:
                    cols[r][i] = round(v, 4)
                    got += 1
            status[i] = "ok" if got == len(rps) else f"partiel : période(s) de retour hors de la courbe ({got}/{len(rps)})"
    for r in rps:
        out[column_name(r)] = cols[r]
    out[f"{RISK}_statut"] = status
    ok = np.array([s == "ok" or s.startswith("partiel") for s in status])
    summary = {"points": len(out), "invalides": int((~valid).sum()), "ok": int(ok.sum()),
               "sans_donnee": int(len(out) - ok.sum() - (~valid).sum() - sum(s.startswith("en attente") for s in status)),
               "pga_max": float(np.nanmax(np.array(list(cols.values())))) if ok.any() else None, "fichier": "service EFEHR (courbes)",
               "rps": rps, "en_attente": sum(x.startswith("en attente") for x in status), "erreurs": errors, "modele": model_id}
    return out, summary


def diagnose_eshm(lat: float = 46.2, lon: float = 6.15, session=None) -> list[dict]:
    """Contrôle pas à pas du service EFEHR sur un point européen (Genève par défaut) ; montre un extrait de chaque réponse brute."""
    rows = []
    try:
        js = _get_json("/v1/models", {"lon": lon, "lat": lat}, session)
        rows.append({"étape": "modèles au point", "résultat": "OK", "détail": str(js)[:400]})
        found = pick_eshm_model(js)
        if found is None:
            rows.append({"étape": "modèle ESHM20", "résultat": "ÉCHEC", "détail": "aucun modèle dont le nom contient « ESHM20 » dans la réponse ci-dessus"})
            return rows
        rows.append({"étape": "modèle ESHM20", "résultat": "OK", "détail": f"identifiant {found[0]} ({found[1]})"})
        js = _get_json("/v1/curves", {"lon": lon, "lat": lat, "modelid": found[0], "imt": "PGA"}, session)
        rows.append({"étape": "courbe PGA (brut)", "résultat": "OK", "détail": str(js)[:500]})
        cs = js.get("curves") if isinstance(js, dict) else None
        if isinstance(cs, list):
            sel_ = pick_curve(cs) if cs and isinstance(cs[0], dict) else {}
            rows.append({"étape": "agrégations proposées", "résultat": "OK",
                         "détail": " ; ".join(f"{c.get('aggregation_type')} {c.get('aggregation_level')}" for c in cs if isinstance(c, dict))
                         + f" — retenue : {sel_.get('aggregation_type')} {sel_.get('aggregation_level')}, durée {sel_.get('investigation_time')} ans"})
        iml, poe = parse_curve(js)
        rows.append({"étape": "lecture de la courbe", "résultat": "OK", "détail": f"{len(iml)} points ; PGA {iml[0]:g} à {iml[-1]:g} ; "
                     f"probabilité {poe[0]:.3g} à {poe[-1]:.3g}"})
        vals = {r: pga_at_rp(iml, poe, r) for r in ESHM_RPS}
        rows.append({"étape": "PGA par période de retour", "résultat": "OK" if vals.get(475) else "À CONTRÔLER",
                     "détail": " ; ".join(f"RP{r} : {'hors courbe' if v is None else round(v, 3)}" for r, v in vals.items())})
    except SeismicError as e:
        rows.append({"étape": "service EFEHR", "résultat": "ÉCHEC", "détail": str(e)})
    return rows


def diagnose(session=None, cache_dir: Path = CACHE_DIR, data_dir: Path = DATA_DIR, raw: bool = False) -> list[dict]:
    rows = []
    cp = compact_path(data_dir)
    if cp:
        try:
            x0, y0, x1, y1 = compact_extent(cp)
            d = _compact(cp)
            v = sample_compact(cp, [6.15, 12.5, 139.7], [46.2, 41.9, 35.7])
            rows.append({"étape": "fichier précalculé", "résultat": "OK", "détail": f"{cp.name} ({cp.stat().st_size / 1e6:.1f} Mo), {d['pga'].shape[1]}×{d['pga'].shape[0]} px, "
                         f"emprise lon {x0:g} à {x1:g}, lat {y0:g} à {y1:g}, précision {1 / float(d['scale']):g} g. PGA à Genève / Rome / Tokyo : "
                         + " ; ".join("aucune valeur" if x is None else f"{x:.3f} g" for x in v) + ". Plus aucun téléchargement n'est nécessaire."})
        except SeismicError as e:
            rows.append({"étape": "fichier précalculé", "résultat": "ÉCHEC", "détail": str(e)})
    if rows and rows[0]["résultat"] == "OK" and not raw:       # fichier précalculé valide : pas de téléchargement du raster brut
        rows.append({"étape": "raster brut", "résultat": "non testé", "détail": "cochez « Données brutes » pour tester aussi le raster d'origine (téléchargement de 35 Mo)."})
        return rows
    try:
        path = gem_path(cache_dir, session, data_dir)
        rows.append({"étape": "fichier GEM", "résultat": "OK", "détail": f"{path.name} ({path.stat().st_size / 1e6:.1f} Mo)"})
    except SeismicError as e:
        if rows and rows[0]["résultat"] == "OK":                # fichier précalculé valide : le raster brut n'est pas nécessaire
            rows.append({"étape": "fichier GEM brut", "résultat": "non lu", "détail": "inutile avec le fichier précalculé (cochez « Données brutes » pour le tester) : " + str(e)[:200]})
            return rows
        return [{"étape": "fichier GEM", "résultat": "ÉCHEC", "détail": str(e)}]
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            p = tf.pages[0]
            rows.append({"étape": "format", "résultat": "OK", "détail": f"{p.imagewidth}×{p.imagelength} px, "
                         f"{'tuilé' if p.is_tiled else 'par bandes'}, compression {p.compression.name}, type {p.dtype}"})
        ext = _extent(path)
        rows.append({"étape": "emprise", "résultat": "OK" if ext else "À CONTRÔLER",
                     "détail": "illisible" if ext is None else f"lon {ext[0]:.1f} à {ext[2]:.1f}, lat {ext[1]:.1f} à {ext[3]:.1f}"})
        checks = {"Istanbul": (29.0, 41.0), "Tokyo": (139.7, 35.7), "Los Angeles": (-118.2, 34.05), "Lima": (-77.0, -12.05), "Paris": (2.35, 48.85)}
        vals = fhz.sample_geotiff(path, [c[0] for c in checks.values()], [c[1] for c in checks.values()])
        high = [v for v in vals[:4] if v is not None]
        rows.append({"étape": "points de contrôle (PGA)", "résultat": "OK" if high and max(high) > 0.1 else "À CONTRÔLER",
                     "détail": " ; ".join(f"{k} : {'aucune valeur' if v is None else round(v, 3)}" for k, v in zip(checks, vals))
                     + " — attendu : de l'ordre de 0,3 à 0,6 pour les 4 premières villes si l'unité est le g, et nettement moins à Paris."})
    except Exception as e:                                      # noqa: BLE001
        rows.append({"étape": "lecture", "résultat": "ÉCHEC", "détail": f"{type(e).__name__}: {str(e)[:200]}"})
    return rows


# ───────────────────────────── vulnérabilité ─────────────────────────────
def vulnerability_curve(theta_g: float = 0.6, beta: float = 0.6, pga_max: float = 2.0) -> Curve:
    """Taux de dommage lognormal : ``Φ(ln(PGA/θ)/β)`` ; ``θ`` = PGA (g) donnant 50 % de dommage, ``β`` = dispersion. Paramètres à calibrer."""
    if theta_g <= 0 or beta <= 0:
        raise ValueError("θ et β doivent être positifs.")
    x = np.round(np.arange(0.0, float(pga_max) + 0.005, 0.01), 3)
    with np.errstate(divide="ignore"):
        z = np.where(x > 0, np.log(np.where(x > 0, x, 1.0) / theta_g) / beta, -np.inf)
    ratio = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) if np.isfinite(v) else 0.0 for v in z])
    return Curve(x, ratio, f"Lognormale : PGA de mi-dommage θ = {theta_g:g} g, dispersion β = {beta:g} (paramètres illustratifs)")


# ───────────────────────────── service ESHM20 (découverte) ─────────────────────────────
def fetch_openapi(url: str = ESHM_OPENAPI, session=None) -> dict:
    try:
        r = (session or __import__("requests")).get(url, timeout=30, headers={"User-Agent": "CatMap", "Accept": "application/json"})
    except Exception as e:                                      # noqa: BLE001
        raise SeismicError(f"service EFEHR inaccessible ({type(e).__name__}: {str(e)[:150]})") from e
    if r.status_code != 200:
        raise SeismicError(f"service EFEHR : HTTP {r.status_code} ({url})")
    try:
        return r.json()
    except ValueError as e:
        raise SeismicError("réponse non JSON.") from e


def summarize_openapi(spec: dict) -> list[dict]:
    rows = []
    for path, ops in (spec.get("paths") or {}).items():
        for method, op in (ops or {}).items():
            if not isinstance(op, dict):
                continue
            for p in op.get("parameters", []) or []:
                sch = p.get("schema") or {}
                rows.append({"route": f"{method.upper()} {path}", "paramètre": p.get("name"), "type": sch.get("type", ""),
                             "obligatoire": bool(p.get("required")), "valeurs": ", ".join(map(str, sch.get("enum", [])))[:150],
                             "description": str(p.get("description") or "")[:150]})
            if not op.get("parameters"):
                rows.append({"route": f"{method.upper()} {path}", "paramètre": "", "type": "", "obligatoire": False, "valeurs": "",
                             "description": str(op.get("summary") or "")[:150]})
    return rows


def load_eshm_curves(cache_dir: Path = CACHE_DIR) -> dict:
    """Courbes d'aléa ESHM20 déjà obtenues (cache disque) : ``{"lat,lon": (iml, poe)}``."""
    import json as _json
    f = Path(cache_dir) / "eshm20_curves.json"
    if not f.exists():
        return {}
    try:
        store = _json.loads(f.read_text())
    except ValueError:
        return {}
    return {k: (np.array(v["iml"], float), np.array(v["poe"], float)) for k, v in store.items()
            if isinstance(v, dict) and "iml" in v and "poe" in v}


def curve_vs_rp(iml: np.ndarray, poe: np.ndarray) -> pd.DataFrame:
    """Courbe d'aléa en (période de retour, PGA) : ``RP = −1 / ln(1 − poe)`` (poe = probabilité annuelle de dépassement)."""
    m = (iml > 0) & (poe > 0) & (poe < 1)
    rp = -1.0 / np.log1p(-poe[m])
    return pd.DataFrame({"rp": rp, "pga": iml[m]}).sort_values("rp").reset_index(drop=True)
