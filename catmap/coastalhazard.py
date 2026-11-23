"""Hauteur d'eau de submersion marine par point : cartes mondiales WRI Aqueduct Floods (v2, coastal), environ 1 km.

Fichiers : ``https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/<nom>.tif``, avec
``inuncoast_<historical|rcp4p5|rcp8p5>_<nosub|wtsub>_<hist|2030|2050|2080>_rp<NNNN>_<0|…>.tif`` (profondeur en mètres ; ``wtsub`` = avec
subsidence, ``nosub`` = sans). Périodes de retour observées dans la liste ``tiffs.txt`` du dépôt GitHub nismod/aqueduct : 1,5 ; 2 ; 5 ; 10 ; 25 ;
50 ; 100 ; 250 ; 500 ; 1000 ans.

NON vérifié depuis l'environnement de développement (réseau bloqué) : la taille des fichiers, leur organisation (tuilée ou par bandes, codec) et
le suffixe exact des scénarios futurs (``_0``, ``_0_perc_05``, ``_0_perc_50``…). ``resolve_url`` essaie donc plusieurs noms et ``diagnose`` fournit
un tableau de contrôle. Seul l'historique (``hist``) a un nom confirmé par les exemples fournis par l'utilisateur.

Lecture : même moteur que ``floodhazard`` (téléchargement avec reprise et contrôle d'intégrité, lecture par segments avec ``tifffile``).
Convention : pour un point dans l'image, « aucune valeur » = 0 (aucune submersion modélisée) ; la résolution (~1 km) est grossière à l'échelle d'un
site, et un point à l'intérieur des terres vaut 0.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import floodhazard as fhz

BASE_URL = "https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/"
RISK = "submersion"
RPS = (2, 5, 10, 25, 50, 100, 250, 500, 1000)
SCENARIOS = {"historical": "Historique", "rcp4p5": "RCP 4.5 (émissions modérées)", "rcp8p5": "RCP 8.5 (émissions élevées)"}
YEARS = {"historical": ("hist",), "rcp4p5": ("2030", "2050", "2080"), "rcp8p5": ("2030", "2050", "2080")}
SUBSIDENCE = {"nosub": "Sans subsidence", "wtsub": "Avec subsidence"}
DEFAULT_CACHE = fhz.DEFAULT_CACHE
TileError = fhz.TileError


def column_name(rp: int) -> str:
    """``AAA_BBB_CCC`` : aléa, scénario, variable (ex. ``submersion_RP100_raw``)."""
    return f"{RISK}_RP{int(rp)}_raw"


S3_HOST = "https://wri-projects.s3.amazonaws.com/"
S3_PREFIX = "AqueductFloodTool/download/v2/"


def candidate_names(scenario: str, sub: str, year: str, rp: int) -> list[str]:
    """Noms de fichier plausibles, dans l'ordre d'essai. Les suffixes réels (``_0``, ``_5``, ``_0_perc_50``…) ne sont pas confirmés : la liste est
    large, et ``list_remote_names`` interroge en priorité le listing public du dépôt."""
    base = f"inuncoast_{scenario}_{sub}_{year}_rp{int(rp):04d}"
    if scenario == "historical":
        return [f"{base}_0.tif", f"{base}_5.tif", f"{base}_0_perc_50.tif", f"{base}_0_perc_05.tif", f"{base}_0_perc_95.tif", f"{base}.tif"]
    return [f"{base}_0_perc_50.tif", f"{base}_0.tif", f"{base}_5.tif", f"{base}_0_perc_05.tif", f"{base}_0_perc_95.tif", f"{base}.tif"]


def list_remote_names(scenario: str, sub: str, year: str, rp: int, session=None) -> list[str]:
    """Noms réellement présents dans le dépôt S3 (listing XML public), pour le préfixe du scénario ; liste vide si le listing est indisponible.
    Ordre de préférence : médiane (``_perc_50``), puis ``_0``, puis les autres."""
    import re
    prefix = f"{S3_PREFIX}inuncoast_{scenario}_{sub}_{year}_rp{int(rp):04d}"
    try:
        r = fhz._http_get(session, S3_HOST, params={"prefix": prefix, "max-keys": "50"}, headers={"User-Agent": "CatMap"}, timeout=60)
        if r.status_code != 200:
            return []
        keys = re.findall(r"<Key>([^<]+\.tif)</Key>", r.text)
    except Exception:                                          # noqa: BLE001
        return []
    names = [k.rsplit("/", 1)[-1] for k in keys]
    pref = lambda n: (0 if "perc_50" in n else 1 if n.endswith("_0.tif") else 2, n)   # noqa: E731
    return sorted(names, key=pref)


def resolve_url(scenario: str, sub: str, year: str, rp: int, session=None) -> str:
    """Première URL qui répond (requête partielle d'1 octet) ; ``TileError`` si aucun nom n'existe."""
    tried = []
    listed = list_remote_names(scenario, sub, year, rp, session)
    for name in list(dict.fromkeys(listed + candidate_names(scenario, sub, year, rp))):
        url = BASE_URL + name
        try:
            r = fhz._http_get(session, url, stream=True, headers={"Range": "bytes=0-0", "User-Agent": "CatMap"}, timeout=60)
            code = r.status_code
            try:
                r.close()
            except Exception:                                  # noqa: BLE001
                pass
        except Exception as e:                                 # noqa: BLE001
            tried.append(f"{name} ({type(e).__name__})")
            continue
        if code in (200, 206):
            return url
        tried.append(f"{name} (HTTP {code})")
    raise TileError("fichier introuvable : " + " ; ".join(tried))


def _extent(path: Path) -> Optional[tuple]:
    """Emprise du raster (ouest, sud, est, nord) en degrés, ou ``None`` si illisible."""
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            x0, y0, sx, sy = fhz._georef(page)
            return (x0, y0 - sy * int(page.imagelength), x0 + sx * int(page.imagewidth), y0)
    except Exception:                                          # noqa: BLE001
        return None


def estimate_depths(df: pd.DataFrame, lat_col: str, lon_col: str, rps: list[int], scenario: str = "historical", sub: str = "nosub",
                    year: str = "hist", session=None, cache_dir: Path = DEFAULT_CACHE, max_mb: float = 900.0,
                    on_progress: Optional[Callable[[int, int], None]] = None) -> tuple[pd.DataFrame, dict]:
    """Ajoute ``submersion_RP<rp>_raw`` (m) et ``submersion_statut``. Une période de retour en échec n'arrête pas les autres."""
    rps = [r for r in rps if r in RPS]
    out = df.copy().reset_index(drop=True)
    lat = pd.to_numeric(out[lat_col], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(out[lon_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90) & (abs(lon) <= 180)
    idx = np.nonzero(valid)[0]
    cols = {column_name(rp): [None] * len(out) for rp in rps}
    problems: dict[str, str] = {}
    files: dict[str, str] = {}
    outside = np.zeros(len(out), bool)
    positives: dict[str, int] = {}
    for done, rp in enumerate(rps, start=1):
        name = column_name(rp)
        try:
            url = resolve_url(scenario, sub, year, rp, session)
            path = fhz.download_tile(url, cache_dir, session, max_mb=max_mb)
            files[name] = str(path)
            try:
                vals = fhz.sample_geotiff(path, lon[idx], lat[idx])
            except TileError:                                   # cache abîmé : un seul retéléchargement
                Path(path).unlink(missing_ok=True)
                path = fhz.download_tile(url, cache_dir, session, max_mb=max_mb)
                vals = fhz.sample_geotiff(path, lon[idx], lat[idx])
            ext = _extent(path)
            n_pos = 0
            for i, v in zip(idx, vals):
                if ext is not None and not (ext[0] <= lon[i] <= ext[2] and ext[1] <= lat[i] <= ext[3]):
                    outside[i] = True                           # hors de l'image : on ne met pas 0, la valeur reste vide
                    continue
                cols[name][i] = 0.0 if v is None or not math.isfinite(v) or v < 0 else round(float(v), 3)
                n_pos += cols[name][i] > 0
            positives[f"RP{rp}"] = int(n_pos)
        except TileError as e:
            problems[f"RP{rp}"] = str(e)
        if on_progress:
            on_progress(done, len(rps))
    for name, vals in cols.items():
        out[name] = vals
    status = []
    for i in range(len(out)):
        if not valid[i]:
            status.append("coordonnées invalides")
        elif outside[i]:
            status.append("hors du raster (hors de l'emprise de la carte)")
        elif problems and any(v is None for v in (cols[c][i] for c in cols)):
            status.append("partiel : " + " ; ".join(f"{k} {v}"[:120] for k, v in problems.items()))
        else:
            status.append("ok")
    out[f"{RISK}_statut"] = status
    summary = {"points": len(out), "invalides": int((~valid).sum()), "fichiers": len(rps), "fichiers_lus": files, "problemes": problems, "hors_raster": int(outside.sum()),
               "points_positifs": positives,
               "scenario": scenario, "sous_scenario": sub, "annee": year}
    return out, summary


def diagnose(scenario: str = "historical", sub: str = "nosub", year: str = "hist", rp: int = 100, session=None) -> list[dict]:
    """Tableau de contrôle : nom de fichier trouvé, taille, organisation (tuilé / bandes), codec."""
    rows = []
    try:
        url = resolve_url(scenario, sub, year, rp, session)
        rows.append({"étape": "nom de fichier", "résultat": "OK", "détail": url.rsplit("/", 1)[-1]})
    except TileError as e:
        return [{"étape": "nom de fichier", "résultat": "ÉCHEC", "détail": str(e)}]
    try:
        path = fhz.download_tile(url, DEFAULT_CACHE, session, max_mb=900)
        rows.append({"étape": "téléchargement", "résultat": "OK", "détail": f"{path.stat().st_size / 1e6:.1f} Mo"})
    except TileError as e:
        rows.append({"étape": "téléchargement", "résultat": "ÉCHEC", "détail": str(e)})
        return rows
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            p = tf.pages[0]
            rows.append({"étape": "format", "résultat": "OK", "détail": f"{p.imagewidth}×{p.imagelength} px, "
                         f"{'tuilé' if p.is_tiled else 'par bandes'}, compression {p.compression.name}, type {p.dtype}"})
        ext = _extent(path)
        rows.append({"étape": "emprise", "résultat": "OK" if ext else "À CONTRÔLER",
                     "détail": "illisible" if ext is None else f"lon {ext[0]:.1f} à {ext[2]:.1f}, lat {ext[1]:.1f} à {ext[3]:.1f}"})
        checks = {"Venise (lagune)": (12.3155, 45.4408), "Rotterdam": (4.48, 51.92), "La Nouvelle-Orléans": (-90.07, 29.95),
                  "Paris (intérieur des terres)": (2.35, 48.85)}
        vals = fhz.sample_geotiff(path, [c[0] for c in checks.values()], [c[1] for c in checks.values()])
        rows.append({"étape": "points de contrôle", "résultat": "OK" if any(v for v in vals) else "À CONTRÔLER",
                     "détail": " ; ".join(f"{k} : {'aucune valeur' if v is None else v}" for k, v in zip(checks, vals))
                     + (" (aucune hauteur > 0 sur les côtes testées : lecture à vérifier)" if not any(v for v in vals[:3]) else "")})
    except Exception as e:                                      # noqa: BLE001
        rows.append({"étape": "lecture", "résultat": "ÉCHEC", "détail": f"{type(e).__name__}: {str(e)[:200]}"})
    return rows
