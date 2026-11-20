"""Aléa sismique par point : accélération du sol (PGA, en g) par période de retour.

Jeux de données :

* **GEM Global Seismic Hazard Map v2023.1** (Zenodo 8409647, licence CC BY-NC-SA 4.0 : usage non commercial). Monde, un seul niveau : PGA
  à 10 % de probabilité de dépassement en 50 ans (période de retour 475 ans), sol rocheux (Vs30 760–800 m/s), raster ``v2023_1_pga_475_rock_3min.tif``
  (3 minutes d'arc ≈ 5 km). Le zip (≈ 35 Mo) est téléchargé à la demande et le raster extrait dans le cache ; un fichier déjà placé dans
  ``data/seisme/`` est utilisé en priorité (le raster de 173 Mo dépasse la limite des dépôts GitHub classiques).
* **ESHM20 (Europe, EFEHR, CC BY 4.0)** : accès par service web ; paramètres non confirmés, voir ``fetch_openapi`` / ``summarize_openapi``.

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
DATASETS = {"gem": {"label": "GEM – carte mondiale (475 ans)", "rps": GEM_RPS},
            "eshm20": {"label": "ESHM20 – Europe (plusieurs périodes de retour)", "rps": ()}}
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
    if not (zpath.exists() and zipfile.is_zipfile(zpath)):
        tmp = zpath.with_suffix(".part")
        try:
            r = (session or __import__("requests")).get(url, stream=True, timeout=120, headers={"User-Agent": "CatMap"})
            if r.status_code != 200:
                raise SeismicError(f"téléchargement GEM : HTTP {r.status_code} ({url})")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        except SeismicError:
            raise
        except Exception as e:                                   # noqa: BLE001
            raise SeismicError(f"téléchargement GEM impossible ({type(e).__name__}: {str(e)[:150]})") from e
        if not zipfile.is_zipfile(tmp):
            tmp.unlink(missing_ok=True)
            raise SeismicError("le fichier téléchargé n'est pas une archive zip valide.")
        tmp.replace(zpath)
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


def estimate_pga(df: pd.DataFrame, lat_col: str, lon_col: str, dataset: str = "gem", rps=GEM_RPS, session=None,
                 cache_dir: Path = CACHE_DIR, tif: Optional[Path] = None) -> tuple[pd.DataFrame, dict]:
    """Ajoute ``seisme_RP475_pga`` (g) et ``seisme_statut``. Les rps non disponibles pour le jeu sont ignorés et signalés."""
    if dataset != "gem":
        raise SeismicError("seul le jeu GEM est branché pour l'instant ; ESHM20 attend la confirmation des paramètres de son service web.")
    rps = [r for r in rps if r in GEM_RPS]
    if not rps:
        raise SeismicError("aucune période de retour disponible pour ce jeu (GEM : 475 ans uniquement).")
    out = df.copy().reset_index(drop=True)
    num = lambda c: pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False), errors="coerce").to_numpy(float)  # noqa: E731
    lat, lon = num(lat_col), num(lon_col)
    valid = np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90) & (abs(lon) <= 180)
    idx = np.nonzero(valid)[0]
    path = Path(tif) if tif else gem_path(cache_dir, session)
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
               "pga_max": float(np.nanmax(col)) if ok.any() else None, "fichier": str(path), "rps": rps}
    return out, summary


def diagnose(session=None, cache_dir: Path = CACHE_DIR) -> list[dict]:
    rows = []
    try:
        path = gem_path(cache_dir, session)
        rows.append({"étape": "fichier GEM", "résultat": "OK", "détail": f"{path.name} ({path.stat().st_size / 1e6:.1f} Mo)"})
    except SeismicError as e:
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
