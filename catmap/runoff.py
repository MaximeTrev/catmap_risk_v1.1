"""Susceptibilité au ruissellement pluvial par point : simplification de la méthode IRIP (Indicateur de Ruissellement Intense Pluvial, Irstea / INRAE).

IRIP : trois cartes (production, transfert, accumulation) de 5 facteurs binaires (0 = défavorable, 1 = favorable au ruissellement), chaque facteur étant noté
d'après une analyse statistique de ses valeurs ; score 0 à 5 ; aucune pluie en entrée ; méthode qualitative (ni débit, ni hauteur d'eau).

Simplification CatMap (à lire avec ``references.py`` et l'infobulle de l'application) :
* données mondiales libres lues dans Earth Engine : MERIT Hydro (altitude et aire drainée amont, 90 m), ESA WorldCover (occupation du sol, 10 m),
  GCN250 (numéro de courbe SCS qui combine occupation du sol et groupe hydrologique de sol, 250 m) ;
* facteurs notés 0/1 par comparaison de la valeur du pixel au quartile de la valeur dans une fenêtre locale (rayon ``radius_km``), à l'image de
  l'analyse statistique de l'IRIP (les seuils exacts de l'IRIP ne sont pas repris : ils sont dans la note technique de Dehotin et Breil, 2011) ;
* profondeur et érodabilité du sol, compacité, axes linéaires (routes, voies ferrées) ne sont pas utilisés ;
* chaque carte est ramenée à [0, 1] (facteurs favorables / facteurs disponibles) puis la susceptibilité globale est la moyenne des trois (poids égaux).

NON testé en réel depuis l'environnement de développement (réseau bloqué) : les identifiants des jeux Earth Engine et leurs bandes sont à confirmer par
``diagnose``. Un jeu absent est écarté (le score se calcule sur les facteurs disponibles) et signalé.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import coastalhazard as ch

RISK = "ruissellement"
TileError = ch.TileError
MERIT_ASSET = "MERIT/Hydro/v1_0_1"                 # bandes attendues : elv (m), upa (km²)
WORLDCOVER_ASSET = "ESA/WorldCover/v200"          # collection ; bande « Map »
GCN_ASSET = "users/jaafarhadi/GCN250/GCN250Average"   # image ; 1re bande = numéro de courbe (condition d'humidité moyenne)
RUNOFF_CLASSES = (40, 50, 60)                      # WorldCover : cultures, bâti, sol nu (peu d'interception et d'infiltration)
SCALE_M = 90
P_HIGH, P_LOW = 75, 25
LEVELS = {1: "très faible", 2: "faible", 3: "modérée", 4: "forte", 5: "très forte"}
COLORS = {1: "#2e7d32", 2: "#9ccc65", 3: "#fdd835", 4: "#fb8c00", 5: "#c62828"}
CHUNK = 100
WORKERS = 4
RADII_KM = (1.0, 2.0, 5.0)
COLUMNS = {"prod": f"{RISK}_susc_production", "transfert": f"{RISK}_susc_transfert", "accum": f"{RISK}_susc_accumulation",
           "globale": f"{RISK}_susc_globale", "classe": f"{RISK}_susc_classe"}
RAW = {"slope": f"{RISK}_pente_deg", "upa": f"{RISK}_aire_drainee_km2", "twi": f"{RISK}_indice_topo", "cn": f"{RISK}_cn",
       "lcfav": f"{RISK}_part_sol_favorable"}
STATUS = f"{RISK}_statut"
_LAYER_OK: dict = {}


# ───────────────────────────── notation (Python pur, testable sans Earth Engine) ─────────────────────────────
def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _gt(v, thr) -> Optional[int]:
    v, thr = _num(v), _num(thr)
    return None if v is None or thr is None else int(v > thr)


def _lt(v, thr) -> Optional[int]:
    v, thr = _num(v), _num(thr)
    return None if v is None or thr is None else int(v < thr)


def _mean(flags: list) -> Optional[float]:
    ok = [f for f in flags if f is not None]
    return None if not ok else sum(ok) / len(ok)


def score_point(v: dict, p: dict) -> dict:
    """Scores de production, transfert, accumulation (chacun dans [0, 1]) et globale (moyenne des trois), classe 1 à 5.
    ``v`` : valeurs au pixel (slope, conc, twi, upa, lcfav, cn) ; ``p`` : quartiles locaux (``<bande>_p25`` / ``<bande>_p75``).
    Un facteur sans donnée est écarté ; une carte sans aucun facteur donne ``None``."""
    f_slope = _gt(v.get("slope"), p.get("slope_p75"))                  # pente forte : favorable à la production et au transfert
    lc = _num(v.get("lcfav"))
    f_lc = None if lc is None else int(lc >= 0.5)                      # sol cultivé, bâti ou nu sur au moins la moitié du pixel
    f_cn = _gt(v.get("cn"), p.get("cn_p75"))                           # numéro de courbe fort : peu d'infiltration
    prod = _mean([f_slope, f_lc, f_cn])
    f_prod = None if prod is None else int(prod >= 0.5)
    f_upa = _gt(v.get("upa"), p.get("upa_p75"))                        # concentration de l'écoulement (talweg)
    transfer = _mean([f_prod, f_slope, f_upa])
    accum = _mean([f_prod, f_upa, _lt(v.get("slope"), p.get("slope_p25")), _gt(v.get("conc"), p.get("conc_p75")), _gt(v.get("twi"), p.get("twi_p75"))])
    maps = (prod, transfer, accum)
    glob = None if any(m is None for m in maps) else sum(maps) / 3.0
    cls = None if glob is None else 1 + min(4, int(glob * 5))
    return {"prod": prod, "transfert": transfer, "accum": accum, "globale": glob, "classe": cls}


# ───────────────────────────── Earth Engine ─────────────────────────────
def layers_available(ee) -> dict:
    """Jeux optionnels lisibles (mis en cache) : ``{"worldcover": bool, "gcn": bool}``. MERIT Hydro est indispensable."""
    if _LAYER_OK:
        return dict(_LAYER_OK)
    try:
        bands = ee.Image(MERIT_ASSET).bandNames().getInfo()
    except Exception as e:                                      # noqa: BLE001
        raise TileError(f"MERIT Hydro illisible ({type(e).__name__}: {str(e)[:200]})") from e
    if "elv" not in bands or "upa" not in bands:
        raise TileError(f"MERIT Hydro : bandes elv/upa absentes (bandes trouvées : {bands})")
    for key, getter in (("worldcover", lambda: ee.ImageCollection(WORLDCOVER_ASSET).first().bandNames().getInfo()),
                        ("gcn", lambda: ee.Image(GCN_ASSET).bandNames().getInfo())):
        try:
            _LAYER_OK[key] = bool(getter())
        except Exception:                                       # noqa: BLE001
            _LAYER_OK[key] = False
    return dict(_LAYER_OK)


def build_layers(ee, use: dict):
    """Image multibande : pente (°), concavité (m), indice topographique ln(a/tanβ), aire drainée (km²), et, si disponibles, part de pixels de sol
    favorable (WorldCover) et numéro de courbe (GCN250). Tout est ramené à la grille de MERIT Hydro (90 m)."""
    mh = ee.Image(MERIT_ASSET)
    elv, upa = mh.select("elv"), mh.select("upa")
    slope = ee.Terrain.slope(elv).rename("slope")
    conc = elv.focalMean(radius=1, kernelType="square", units="pixels").subtract(elv).rename("conc")   # > 0 : le pixel est plus bas que ses voisins
    side = ee.Image.pixelArea().sqrt()
    tan_b = slope.multiply(math.pi / 180.0).tan().max(0.001)
    twi = upa.multiply(1e6).divide(side).divide(tan_b).log().rename("twi")
    bands = [slope, conc, twi, upa.rename("upa")]
    proj = elv.projection()
    if use.get("worldcover"):
        lc = ee.ImageCollection(WORLDCOVER_ASSET).first().select("Map")
        fav = lc.eq(RUNOFF_CLASSES[0]).Or(lc.eq(RUNOFF_CLASSES[1])).Or(lc.eq(RUNOFF_CLASSES[2])).rename("lcfav")
        bands.append(fav.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(proj))
    if use.get("gcn"):
        bands.append(ee.Image(GCN_ASSET).select(0).rename("cn").resample("bilinear").reproject(proj))
    return ee.Image.cat(bands).updateMask(elv.mask())


def _reduce_chunk(ee, layers, pts: list, radius_km: float, names: list) -> dict:
    """``pts`` : [(clé, lon, lat)] -> {clé: (valeurs au pixel, quartiles locaux)} ; les points sans valeur (mer, hors couverture) ont des valeurs vides."""
    feats_pt = [ee.Feature(ee.Geometry.Point([lo, la]), {"k": int(k)}) for k, lo, la in pts]
    feats_win = [ee.Feature(ee.Geometry.Point([lo, la]).buffer(radius_km * 1000.0), {"k": int(k)}) for k, lo, la in pts]
    at = layers.reduceRegions(collection=ee.FeatureCollection(feats_pt), reducer=ee.Reducer.first().forEach(names), scale=SCALE_M).getInfo()
    pc = ee.Reducer.percentile([P_LOW, P_HIGH])
    win = layers.select([n for n in names if n != "lcfav"]).reduceRegions(collection=ee.FeatureCollection(feats_win), reducer=pc, scale=SCALE_M,
                                                                            tileScale=4).getInfo()
    out: dict = {int(k): ({}, {}) for k, _, _ in pts}
    for f in at.get("features", []):
        pr = f.get("properties", {})
        out[int(pr["k"])][0].update({n: pr.get(n) for n in names})
    for f in win.get("features", []):
        pr = f.get("properties", {})
        out[int(pr["k"])][1].update({k: v for k, v in pr.items() if k != "k"})
    return out


def estimate(df: pd.DataFrame, lat_col: str, lon_col: str, radius_km: float = 2.0, ee_project: Optional[str] = None, ee_account=None,
             on_progress: Optional[Callable[[int, int], None]] = None, chunk: int = CHUNK, workers: int = WORKERS) -> tuple[pd.DataFrame, dict]:
    """Ajoute les colonnes de ``COLUMNS``, ``RAW`` et ``STATUS``. Les points au même pixel (coordonnées arrondies à 1e-4°) sont calculés une seule
    fois, les lots sont traités en parallèle."""
    if radius_km not in RADII_KM and not 0 < radius_km <= 20:
        raise ValueError("rayon de fenêtre locale invalide (0 à 20 km).")
    out = df.copy().reset_index(drop=True)
    lat = pd.to_numeric(out[lat_col], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(out[lon_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90) & (abs(lon) <= 180)
    ch.init_ee(ee_project, ee_account)
    ee = ch._ee()
    use = layers_available(ee)
    names = ["slope", "conc", "twi", "upa"] + (["lcfav"] if use.get("worldcover") else []) + (["cn"] if use.get("gcn") else [])
    layers = build_layers(ee, use)
    keys: dict = {}
    for i in np.nonzero(valid)[0]:
        keys.setdefault((round(float(lat[i]), 4), round(float(lon[i]), 4)), []).append(int(i))
    uniq = list(keys)
    batches = [[(j, uniq[j][1], uniq[j][0]) for j in range(a, min(len(uniq), a + chunk))] for a in range(0, len(uniq), chunk)]
    results: dict = {}
    failed: dict = {}

    def work(b):
        return _reduce_chunk(ee, layers, b, float(radius_km), names)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(work, b): b for b in batches}
        for fu in as_completed(futs):
            b = futs[fu]
            try:
                results.update(fu.result())
            except Exception as e:                              # noqa: BLE001
                for k, _, _ in b:
                    failed[k] = f"{type(e).__name__}: {str(e)[:160]}"
            done += 1
            if on_progress:
                on_progress(done, len(batches))
    cols = {c: [None] * len(out) for c in list(COLUMNS.values()) + list(RAW.values())}
    status = ["ok"] * len(out)
    n_sea = 0
    for j, (key, idxs) in enumerate(keys.items()):
        if j in failed:
            for i in idxs:
                status[i] = "échec Earth Engine : " + failed[j]
            continue
        v, p = results.get(j, ({}, {}))
        if _num(v.get("upa")) is None or _num(v.get("slope")) is None:
            n_sea += len(idxs)
            for i in idxs:
                status[i] = "hors terre ou hors couverture (MERIT Hydro : 90°N à 60°S, terres émergées)"
            continue
        sc = score_point(v, p)
        for k, c in COLUMNS.items():
            for i in idxs:
                cols[c][i] = sc[k]
        for k, c in RAW.items():
            for i in idxs:
                cols[c][i] = _num(v.get(k))
    for i in range(len(out)):
        if not valid[i]:
            status[i] = "coordonnées invalides"
    for c, vals in cols.items():
        out[c] = vals
    out[STATUS] = status
    classes = pd.to_numeric(out[COLUMNS["classe"]], errors="coerce")
    summary = {"points": len(out), "invalides": int((~valid).sum()), "pixels_uniques": len(uniq), "lots": len(batches), "echecs": len(failed),
               "hors_terre": n_sea, "facteurs": names, "jeux_absents": [k for k, v in use.items() if not v], "rayon_km": radius_km,
               "repartition": {LEVELS[int(k)]: int(v) for k, v in classes.value_counts().sort_index().items()}}
    return out, summary


def diagnose(ee_project: Optional[str], ee_account, radius_km: float = 2.0) -> list[dict]:
    """Tableau de contrôle : secrets, authentification, jeux de données, bandes, puis notation sur des lieux connus."""
    rows = []
    try:
        ch.init_ee(ee_project, ee_account)
        rows.append({"étape": "authentification", "résultat": "OK", "détail": "Earth Engine initialisé"})
    except TileError as e:
        return [{"étape": "authentification", "résultat": "ÉCHEC", "détail": str(e)}]
    ee = ch._ee()
    try:
        _LAYER_OK.clear()
        use = layers_available(ee)
        rows.append({"étape": "MERIT Hydro", "résultat": "OK", "détail": f"{MERIT_ASSET} : bandes elv et upa présentes"})
        for key, label in (("worldcover", f"ESA WorldCover ({WORLDCOVER_ASSET})"), ("gcn", f"GCN250 ({GCN_ASSET})")):
            rows.append({"étape": label, "résultat": "OK" if use.get(key) else "ABSENT", "détail":
                         "lisible" if use.get(key) else "jeu illisible : le facteur correspondant est écarté du score"})
        places = {"Vallon des Combes, Biot (France)": (7.10, 43.62), "Alpes, Chamonix": (6.87, 45.92), "Paris": (2.35, 48.85),
                  "Delta du Gange": (90.4, 22.3), "Atlantique (mer)": (-30.0, 40.0)}
        df = pd.DataFrame({"lon": [c[0] for c in places.values()], "lat": [c[1] for c in places.values()]})
        res, s = estimate(df, "lat", "lon", radius_km, ee_project, ee_account, workers=1)
        for name, (_, r) in zip(places, res.iterrows()):
            if r[STATUS] != "ok":
                rows.append({"étape": name, "résultat": "À CONTRÔLER", "détail": r[STATUS]})
            else:
                rows.append({"étape": name, "résultat": "OK", "détail":
                             f"pente {r[RAW['slope']]:.1f}° ; aire drainée {r[RAW['upa']]:.1f} km² ; production {r[COLUMNS['prod']]:.2f} ; transfert "
                             f"{r[COLUMNS['transfert']]:.2f} ; accumulation {r[COLUMNS['accum']]:.2f} ; globale {r[COLUMNS['globale']]:.2f} "
                             f"(classe {int(r[COLUMNS['classe']])})"})
    except TileError as e:
        rows.append({"étape": "lecture", "résultat": "ÉCHEC", "détail": str(e)})
    except Exception as e:                                      # noqa: BLE001
        rows.append({"étape": "lecture", "résultat": "ÉCHEC", "détail": f"{type(e).__name__}: {str(e)[:220]}"})
    return rows
