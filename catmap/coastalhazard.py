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


def candidate_names(scenario: str, sub: str, year: str, rp: int) -> list[str]:
    """Noms de fichier plausibles, dans l'ordre d'essai (l'historique est confirmé ; les futurs sont à valider par ``resolve_url``)."""
    base = f"inuncoast_{scenario}_{sub}_{year}_rp{int(rp):04d}"
    if scenario == "historical":
        return [f"{base}_0.tif"]
    return [f"{base}_0_perc_50.tif", f"{base}_0.tif", f"{base}_0_perc_05.tif", f"{base}_0_perc_95.tif"]


def resolve_url(scenario: str, sub: str, year: str, rp: int, session=None) -> str:
    """Première URL qui répond (requête partielle d'1 octet) ; ``TileError`` si aucun nom n'existe."""
    tried = []
    for name in candidate_names(scenario, sub, year, rp):
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
            for i, v in zip(idx, vals):
                cols[name][i] = 0.0 if v is None or not math.isfinite(v) or v < 0 else round(float(v), 3)
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
        elif problems and any(v is None for v in (cols[c][i] for c in cols)):
            status.append("partiel : " + " ; ".join(f"{k} {v}"[:120] for k, v in problems.items()))
        else:
            status.append("ok")
    out[f"{RISK}_statut"] = status
    summary = {"points": len(out), "invalides": int((~valid).sum()), "fichiers": len(rps), "fichiers_lus": files, "problemes": problems,
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
        v = fhz.sample_geotiff(path, [2.35], [48.85])
        rows.append({"étape": "lecture d'un point", "résultat": "OK", "détail": f"Paris : {v[0]}"})
    except Exception as e:                                      # noqa: BLE001
        rows.append({"étape": "lecture", "résultat": "ÉCHEC", "détail": f"{type(e).__name__}: {str(e)[:200]}"})
    return rows
