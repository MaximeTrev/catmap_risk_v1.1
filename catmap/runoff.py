"""Susceptibilité au ruissellement pluvial par point : simplification de la méthode IRIP (Indicateur de Ruissellement Intense Pluvial, Irstea / INRAE).

IRIP : trois cartes (production, transfert, accumulation) de facteurs binaires (0 = défavorable, 1 = favorable au ruissellement) sommés en un score de 0 à
5 ; aucune pluie en entrée ; méthode qualitative (ni débit, ni hauteur d'eau). Dans l'application de Braud et al. (2020, NHESS 20, 947), les seuils de pente
(T1) et d'indice topographique (T2) sont calculés pour chaque zone d'étude par une classification en 2 classes (Rubin, 1967 ; type k-means), les autres
seuils étant surtout absolus (aire drainée 2,5 ha, rupture de pente ±0,0018 sur un MNT de 25 m, Ks < 1e-6 m/s, épaisseur du sol < 50 cm…).

Simplification CatMap (à lire avec ``references.py`` et l'infobulle de l'application) :
* données mondiales libres lues dans Earth Engine : MERIT Hydro (altitude, aire drainée amont ; 90 m), ESA WorldCover (occupation du sol), et au choix pour le
  sol GCN250 (numéro de courbe), HiHydroSoil v2.0 (groupe hydrologique de sol ou Ks) ; en option les géomorphons de Geomorpho90m (Barjot et al., 2023) ;
* seuils calculés pour chaque **zone**, bassin versant approché par l'unité de drainage HydroBASINS (niveau au choix) ou disque autour du point (rayon au
  choix), par la même classification en 2 classes (optimum de la variance intra-classe en 1 dimension, équivalent à un k-means à 2 classes) ;
* rupture de pente : courbure du profil (Zevenbergen et Thorne, 1987) sur la grille de 90 m, seuil = amplitude séparant 2 classes dans la zone (le seuil de
  0,0018 de l'IRIP dépend de la fonction GRASS et de la résolution du MNT : non transposable) ;
* érodabilité, épaisseur du sol, compacité, axes linéaires non utilisés ; production amont = production du pixel du point (≥ 3/5 soit ≥ 0,6), faute de
  pouvoir agréger sur tout le sous-bassin amont ;
* chaque carte est ramenée à [0, 1] (somme des facteurs favorables / somme des facteurs disponibles) ; globale = moyenne des trois (poids égaux).

NON testé en réel depuis l'environnement de développement (réseau bloqué) : les identifiants des jeux Earth Engine, leurs bandes et échelles sont à confirmer
par ``diagnose``. Un jeu absent est écarté (le score se calcule sur les facteurs disponibles) et signalé.
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
HIHYDRO_BASE = "projects/sat-io/open-datasets/HiHydroSoilv2_0/"
HSG_ASSET = HIHYDRO_BASE + "Hydrologic_Soil_Group_250m"   # 1=A 2=B 3=C 4=D 14=A/D 24=B/D 34=C/D
KSAT_ASSET = HIHYDRO_BASE + "ksat"                # cm/j, stocké × 10 000 selon la page du catalogue communautaire
GEOM_ASSET = "projects/sat-io/open-datasets/Geomorpho90m/geom"   # géomorphons, collection de tuiles, classes 1 à 10
BASINS_ASSET = "WWF/HydroSHEDS/v1/Basins/hybas_{level}"
RUNOFF_CLASSES = (40, 50)                          # WorldCover : cultures, zones bâties (« zones urbaines et terres agricoles » de l'IRIP)
KSAT_SCALE = 1e-4
KSAT_THRESHOLD_CM_D = 8.64                         # 1e-6 m/s (Braud et al., 2020) = 8,64 cm/j
HSG_FAVOURABLE = (4, 14, 24, 34)                   # groupe D, ou double groupe (D à l'état naturel)
AREA_FIXED_KM2 = 0.025                             # 2,5 ha (Braud et al., 2020, d'après Lagadec 2017)
UPSTREAM_PROD_MIN = 0.6                            # « production amont ≥ 3 » sur une échelle de 5
GEOM_SCORE = {2: 0, 3: 0, 6: 0,                    # pic, crête, pente
              1: 1, 5: 1, 4: 1,                    # plat, éperon, épaulement
              7: 2, 8: 2,                          # creux, bas de versant
              9: 3, 10: 3}                         # vallée, fosse (Barjot et al., 2023, tableau 1)
SCALE_M = 90
BASIN_LEVELS = (9, 10, 11, 12)
DEFAULT_LEVEL = 11
WINDOW_KM = (1.0, 2.0, 5.0, 10.0)
SOIL_METHODS = {"cn": "Numéro de courbe (GCN250), seuil par zone", "hsg": "Groupe hydrologique de sol (HiHydroSoil v2.0)",
                "ksat": "Conductivité hydraulique Ks < 1e-6 m/s (HiHydroSoil v2.0)"}
ACCUM_METHODS = {"classic": "Classique : pente, rupture de pente, indice topographique", "geom": "Géomorphons (Barjot et al., 2023)"}
AREA_METHODS = {"zone": "Local : classification de l'aire drainée par zone", "fixed": "Fixe : 2,5 ha (Braud et al., 2020)"}
# Garde-fou : étendue minimale d'une variable dans la zone pour chercher un seuil (sinon la classification en 2 classes sépare du bruit : aucun pixel favorable)
MIN_RANGE = {"slope": 1.0, "twi": 1.0, "lupa": 0.5, "acurv": 1e-4, "cn": 5.0}
MIN_PIXELS = 20
LEVELS = {1: "très faible", 2: "faible", 3: "modérée", 4: "forte", 5: "très forte"}
COLORS = {1: "#2e7d32", 2: "#9ccc65", 3: "#fdd835", 4: "#fb8c00", 5: "#c62828"}
CHUNK_BASIN = 100
CHUNK_WINDOW = 40
WORKERS = 4
COLUMNS = {"prod": f"{RISK}_susc_production", "transfert": f"{RISK}_susc_transfert", "accum": f"{RISK}_susc_accumulation",
           "globale": f"{RISK}_susc_globale", "classe": f"{RISK}_susc_classe", "irip": f"{RISK}_score_irip", "irip3": f"{RISK}_irip_ge3"}
RAW = {"slope": f"{RISK}_pente_deg", "upa": f"{RISK}_aire_drainee_km2", "twi": f"{RISK}_indice_topo", "curv": f"{RISK}_courbure_profil",
       "cn": f"{RISK}_cn", "hsg": f"{RISK}_groupe_hydro", "ksat": f"{RISK}_ksat_cm_j", "geom": f"{RISK}_geomorphon", "lcfav": f"{RISK}_part_sol_favorable"}
ZONE = {"id": f"{RISK}_zone_id", "km2": f"{RISK}_zone_km2", "t_slope": f"{RISK}_seuil_pente_deg", "t_twi": f"{RISK}_seuil_indice_topo",
        "t_area": f"{RISK}_seuil_aire_km2", "t_curv": f"{RISK}_seuil_courbure"}
STATUS = f"{RISK}_statut"
_LAYER_OK: dict = {}
_ASSET_KIND: dict = {}
_BASIN_CACHE: dict = {}


# ───────────────────────────── seuils par zone (Python pur, testable sans Earth Engine) ─────────────────────────────
def otsu_threshold(hist: Optional[dict], min_range: float = 0.0, min_pixels: int = MIN_PIXELS) -> Optional[float]:
    """Seuil séparant les valeurs d'une zone en 2 classes en minimisant la variance intra-classe (en 1 dimension : même optimum qu'un k-means à 2
    classes, ou que la classification de Rubin, 1967), à partir de l'histogramme d'Earth Engine (``bucketMeans``, ``histogram``). ``None`` si la zone
    a trop peu de pixels ou une étendue inférieure à ``min_range``."""
    if not hist or not hist.get("bucketMeans") or not hist.get("histogram"):
        return None
    x = np.asarray(hist["bucketMeans"], float)
    w = np.asarray(hist["histogram"], float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) < 2 or w.sum() < min_pixels:
        return None
    order = np.argsort(x)
    x, w = x[order], w[order]
    if x[-1] - x[0] < min_range:
        return None
    c, cm = np.cumsum(w), np.cumsum(w * x)
    total = c[-1]
    best, best_k = -1.0, 0
    for k in range(len(x) - 1):
        w0, w1 = c[k], total - c[k]
        m0, m1 = cm[k] / w0, (cm[-1] - cm[k]) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, best_k = between, k
    return float((x[best_k] + x[best_k + 1]) / 2.0)


def zone_thresholds(hists: dict) -> dict:
    """Seuils d'une zone à partir des histogrammes de ses variables. Une zone sans seuil (trop petite ou trop uniforme) reçoit +inf : aucun pixel n'y est
    « au-dessus du seuil » (donc non favorable), et tout pixel y est « sous le seuil » (pente faible : favorable à l'accumulation)."""
    inf = float("inf")
    out = {}
    for band, key in (("slope", "slope"), ("twi", "twi"), ("lupa", "lupa"), ("acurv", "curv"), ("cn", "cn")):
        if band not in hists:
            continue
        t = otsu_threshold(hists[band], MIN_RANGE.get(band, 0.0))
        out[key] = inf if t is None else t
    return out


# ───────────────────────────── notation (Python pur) ─────────────────────────────
def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _gt(v, thr) -> Optional[int]:
    v = _num(v)
    return None if v is None or thr is None else int(v > thr)


def _le(v, thr) -> Optional[int]:
    v = _num(v)
    return None if v is None or thr is None else int(v <= thr)


def _lt(v, thr) -> Optional[int]:
    v = _num(v)
    return None if v is None or thr is None else int(v < thr)


def _wmean(pairs: list) -> Optional[float]:
    """Moyenne pondérée des facteurs disponibles ; ``pairs`` : [(valeur ou None, poids maximal)]."""
    ok = [(v, w) for v, w in pairs if v is not None]
    return None if not ok else sum(v for v, _ in ok) / sum(w for _, w in ok)


def soil_factor(v: dict, t: dict, soil: str) -> Optional[int]:
    if soil == "cn":
        return _gt(v.get("cn"), t.get("cn"))
    if soil == "hsg":
        h = _num(v.get("hsg"))
        return None if h is None else int(int(round(h)) in HSG_FAVOURABLE)
    if soil == "ksat":
        return _lt(v.get("ksat"), KSAT_THRESHOLD_CM_D)
    raise ValueError(f"méthode de sol inconnue : {soil}")


def score_point(v: dict, t: dict, soil: str = "cn", accum: str = "classic", area: str = "zone") -> dict:
    """Scores de production, transfert, accumulation (chacun dans [0, 1]), globale (moyenne des trois), classe 1 à 5, score IRIP-équivalent (0 à 5).
    ``v`` : valeurs au pixel ; ``t`` : seuils de la zone (``zone_thresholds``). Un facteur sans donnée est écarté, une carte sans facteur donne ``None``."""
    slope, twi, curv = v.get("slope"), v.get("twi"), v.get("curv")
    a, b = _gt(slope, t.get("slope")), _gt(twi, t.get("twi"))
    topo = None if a is None and b is None else int(bool(a) or bool(b))          # IRIP : pente > T1 ou indice topographique > T2
    lc = _num(v.get("lcfav"))
    f_lc = None if lc is None else int(lc >= 0.5)
    prod = _wmean([(topo, 1), (f_lc, 1), (soil_factor(v, t, soil), 1)])
    f_prod = None if prod is None else int(prod >= UPSTREAM_PROD_MIN)
    f_area = _gt(v.get("upa"), AREA_FIXED_KM2) if area == "fixed" else _gt(v.get("lupa"), t.get("lupa"))
    c = t.get("curv")
    transfer = _wmean([(f_prod, 1), (a, 1), (_lt(curv, None if c is None else -c), 1), (f_area, 1)])
    if accum == "geom":
        g = _num(v.get("geom"))
        score = GEOM_SCORE.get(int(round(g))) if g is not None else None
        acc = _wmean([(f_prod, 1), (f_area, 1), (score, 3)])
    else:
        acc = _wmean([(f_prod, 1), (f_area, 1), (_le(slope, t.get("slope")), 1), (_gt(curv, c), 1), (_gt(twi, t.get("twi")), 1)])
    maps = (prod, transfer, acc)
    glob = None if any(m is None for m in maps) else sum(maps) / 3.0
    cls = None if glob is None else 1 + min(4, int(glob * 5))
    return {"prod": prod, "transfert": transfer, "accum": acc, "globale": glob, "classe": cls,
            "irip": None if glob is None else round(glob * 5.0, 2), "irip3": None if glob is None else int(glob * 5.0 >= 3.0)}


# ───────────────────────────── Earth Engine ─────────────────────────────
def load_asset(ee, path: str):
    """Image ou collection de tuiles (mosaïque) selon ce que l'asset est réellement ; le type est mémorisé."""
    kind = _ASSET_KIND.get(path)
    if kind is None:
        try:
            ee.Image(path).bandNames().getInfo()
            kind = "image"
        except Exception:                                       # noqa: BLE001
            try:
                ee.ImageCollection(path).first().bandNames().getInfo()
                kind = "collection"
            except Exception as e:                              # noqa: BLE001
                raise TileError(f"jeu {path} illisible ({type(e).__name__}: {str(e)[:160]})") from e
        _ASSET_KIND[path] = kind
    return ee.Image(path) if kind == "image" else ee.ImageCollection(path).mosaic()


def layers_available(ee, needs: set) -> dict:
    """Jeux optionnels demandés et lisibles (mémorisé) ; MERIT Hydro est indispensable."""
    if "merit" not in _LAYER_OK:
        try:
            bands = ee.Image(MERIT_ASSET).bandNames().getInfo()
        except Exception as e:                                  # noqa: BLE001
            raise TileError(f"MERIT Hydro illisible ({type(e).__name__}: {str(e)[:200]})") from e
        if "elv" not in bands or "upa" not in bands:
            raise TileError(f"MERIT Hydro : bandes elv/upa absentes (bandes trouvées : {bands})")
        _LAYER_OK["merit"] = True
    getters = {"worldcover": lambda: ee.ImageCollection(WORLDCOVER_ASSET).first().bandNames().getInfo(),
               "gcn": lambda: ee.Image(GCN_ASSET).bandNames().getInfo(), "hsg": lambda: load_asset(ee, HSG_ASSET).bandNames().getInfo(),
               "ksat": lambda: load_asset(ee, KSAT_ASSET).bandNames().getInfo(), "geom": lambda: load_asset(ee, GEOM_ASSET).bandNames().getInfo()}
    for key in needs:
        if key not in _LAYER_OK:
            try:
                _LAYER_OK[key] = bool(getters[key]())
            except Exception:                                   # noqa: BLE001
                _LAYER_OK[key] = False
    return {k: _LAYER_OK.get(k, False) for k in needs}


def build_layers(ee, use: dict):
    """Image multibande à la grille de MERIT Hydro (90 m) : pente (°), courbure du profil (1/m, > 0 = concave), indice topographique ln(a/tanβ), aire
    drainée (km²) et son logarithme, amplitude de courbure, puis les jeux optionnels demandés."""
    mh = ee.Image(MERIT_ASSET)
    elv, upa = mh.select("elv"), mh.select("upa")
    slope = ee.Terrain.slope(elv).rename("slope")
    side = ee.Image.pixelArea().sqrt()                          # côté de pixel en m (approximation de dx et dy)
    tan_b = slope.multiply(math.pi / 180.0).tan().max(0.001)
    twi = upa.multiply(1e6).divide(side).divide(tan_b).log().rename("twi")
    nb = elv.neighborhoodToBands(ee.Kernel.square(1))
    z = {(x, y): nb.select(f"elv_{x}_{y}") for x in (-1, 0, 1) for y in (-1, 0, 1)}
    d = z[(-1, 0)].add(z[(1, 0)]).divide(2).subtract(z[(0, 0)]).divide(side.pow(2))
    e = z[(0, -1)].add(z[(0, 1)]).divide(2).subtract(z[(0, 0)]).divide(side.pow(2))
    f = z[(1, -1)].add(z[(-1, 1)]).subtract(z[(-1, -1)]).subtract(z[(1, 1)]).divide(side.pow(2).multiply(4))
    g = z[(1, 0)].subtract(z[(-1, 0)]).divide(side.multiply(2))
    h = z[(0, -1)].subtract(z[(0, 1)]).divide(side.multiply(2))
    curv = d.multiply(g.pow(2)).add(e.multiply(h.pow(2))).add(f.multiply(g).multiply(h)).multiply(2).divide(g.pow(2).add(h.pow(2)).add(1e-12))
    curv = curv.rename("curv")                                  # Zevenbergen et Thorne (1987), profil ; positif = concave (ralentissement)
    bands = [slope, curv, twi, upa.rename("upa"), upa.max(1e-6).log10().rename("lupa"), curv.abs().rename("acurv")]
    proj = elv.projection()
    if use.get("worldcover"):
        lc = ee.ImageCollection(WORLDCOVER_ASSET).first().select("Map")
        fav = lc.eq(RUNOFF_CLASSES[0]).Or(lc.eq(RUNOFF_CLASSES[1])).rename("lcfav")
        bands.append(fav.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(proj))
    if use.get("gcn"):
        bands.append(ee.Image(GCN_ASSET).select(0).rename("cn").resample("bilinear").reproject(proj))
    if use.get("hsg"):
        bands.append(load_asset(ee, HSG_ASSET).select(0).rename("hsg"))
    if use.get("ksat"):
        bands.append(load_asset(ee, KSAT_ASSET).select(0).multiply(KSAT_SCALE).rename("ksat"))
    if use.get("geom"):
        bands.append(load_asset(ee, GEOM_ASSET).select(0).rename("geom"))
    return ee.Image.cat(bands).updateMask(elv.mask())


def _point_values(ee, layers, pts: list, names: list) -> dict:
    feats = [ee.Feature(ee.Geometry.Point([lo, la]), {"k": int(k)}) for k, lo, la in pts]
    res = layers.select(names).reduceRegions(collection=ee.FeatureCollection(feats), reducer=ee.Reducer.first().forEach(names), scale=SCALE_M).getInfo()
    out = {int(k): {} for k, _, _ in pts}
    for f in res.get("features", []):
        pr = f.get("properties", {})
        out[int(pr["k"])] = {n: pr.get(n) for n in names}
    return out


def _basin_ids(ee, basins, pts: list) -> dict:
    """{clé du point: (identifiant HYBAS, surface km²)} ; (-1, 0) hors de tout bassin (mer)."""
    feats = [ee.Feature(ee.Geometry.Point([lo, la]), {"k": int(k)}) for k, lo, la in pts]

    def tag(f):
        bs = basins.filterBounds(f.geometry())
        first = ee.Feature(bs.first())
        return f.set({"hid": ee.Algorithms.If(bs.size().gt(0), first.get("HYBAS_ID"), -1),
                      "km2": ee.Algorithms.If(bs.size().gt(0), first.get("SUB_AREA"), 0)})
    info = ee.FeatureCollection(feats).map(tag).getInfo()
    return {int(f["properties"]["k"]): (int(f["properties"].get("hid", -1)), float(f["properties"].get("km2") or 0.0)) for f in info.get("features", [])}


def _hist_zones(ee, layers, fc, hist_names: list, key: str) -> dict:
    """Histogrammes des variables de seuil pour chaque entité de ``fc`` : {valeur de ``key``: {bande: histogramme}}."""
    red = ee.Reducer.histogram(maxBuckets=128).forEach(hist_names)
    res = layers.select(hist_names).reduceRegions(collection=fc, reducer=red, scale=SCALE_M, tileScale=8).getInfo()
    out = {}
    for f in res.get("features", []):
        pr = f.get("properties", {})
        out[int(pr[key])] = {n: pr.get(n) for n in hist_names}
    return out


def _process_batch(ee, layers, basins, zone: dict, pts: list, names: list, hist_names: list) -> dict:
    """{clé du point: (valeurs au pixel, seuils de la zone, identifiant de zone, surface de zone km²)}."""
    values = _point_values(ee, layers, pts, names)
    out: dict = {}
    if zone["kind"] == "window":
        feats = [ee.Feature(ee.Geometry.Point([lo, la]).buffer(zone["radius_km"] * 1000.0), {"k": int(k)}) for k, lo, la in pts]
        hist = _hist_zones(ee, layers, ee.FeatureCollection(feats), hist_names, "k")
        area = math.pi * zone["radius_km"] ** 2
        for k, _, _ in pts:
            out[int(k)] = (values[int(k)], zone_thresholds(hist.get(int(k), {})), int(k), area)
        return out
    ids = _basin_ids(ee, basins, pts)
    ckey = (zone["level"], tuple(hist_names))
    todo = sorted({hid for hid, _ in ids.values() if hid >= 0 and (ckey, hid) not in _BASIN_CACHE})
    if todo:
        fc = basins.filter(ee.Filter.inList("HYBAS_ID", todo))
        hist = _hist_zones(ee, layers, fc, hist_names, "HYBAS_ID")
        for hid in todo:
            _BASIN_CACHE[(ckey, hid)] = zone_thresholds(hist.get(hid, {}))
    for k, _, _ in pts:
        hid, km2 = ids.get(int(k), (-1, 0.0))
        out[int(k)] = (values[int(k)], _BASIN_CACHE.get((ckey, hid), {}), hid, km2)
    return out


def estimate(df: pd.DataFrame, lat_col: str, lon_col: str, zone: Optional[dict] = None, ee_project: Optional[str] = None, ee_account=None,
             on_progress: Optional[Callable[[int, int], None]] = None, soil: str = "cn", accum: str = "classic", area: str = "zone",
             chunk: Optional[int] = None, workers: int = WORKERS) -> tuple[pd.DataFrame, dict]:
    """Ajoute les colonnes de ``COLUMNS``, ``RAW``, ``ZONE`` et ``STATUS``. ``zone`` : ``{"kind": "basin", "level": 11}`` (unité de drainage HydroBASINS)
    ou ``{"kind": "window", "radius_km": 2.0}`` (disque autour du point). Les points au même pixel (coordonnées arrondies à 1e-4°) sont calculés une seule
    fois, les lots sont traités en parallèle, les seuils d'un bassin sont calculés une seule fois."""
    zone = zone or {"kind": "basin", "level": DEFAULT_LEVEL}
    if zone.get("kind") == "basin":
        if int(zone.get("level", 0)) not in range(1, 13):
            raise ValueError("niveau de bassin invalide (1 à 12).")
    elif zone.get("kind") == "window":
        if not 0 < float(zone.get("radius_km", 0)) <= 20:
            raise ValueError("rayon de fenêtre locale invalide (0 à 20 km).")
    else:
        raise ValueError("zone inconnue : 'basin' ou 'window' attendu.")
    if soil not in SOIL_METHODS or accum not in ACCUM_METHODS or area not in AREA_METHODS:
        raise ValueError("méthode inconnue (sol, accumulation ou aire drainée).")
    out = df.copy().reset_index(drop=True)
    lat = pd.to_numeric(out[lat_col], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(out[lon_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(lat) & np.isfinite(lon) & (abs(lat) <= 90) & (abs(lon) <= 180)
    ch.init_ee(ee_project, ee_account)
    ee = ch._ee()
    needs = {"worldcover", soil if soil != "cn" else "gcn"} | ({"geom"} if accum == "geom" else set())
    use = layers_available(ee, needs)
    names = ["slope", "curv", "twi", "upa", "lupa"] + [n for n in ("lcfav", "cn", "hsg", "ksat", "geom") if use.get({"lcfav": "worldcover", "cn": "gcn"}.get(n, n))]
    hist_names = ["slope", "twi", "lupa", "acurv"] + (["cn"] if "cn" in names and soil == "cn" else [])
    layers = build_layers(ee, use)
    basins = ee.FeatureCollection(BASINS_ASSET.format(level=int(zone["level"]))) if zone["kind"] == "basin" else None
    size = chunk or (CHUNK_BASIN if zone["kind"] == "basin" else CHUNK_WINDOW)
    keys: dict = {}
    for i in np.nonzero(valid)[0]:
        keys.setdefault((round(float(lat[i]), 4), round(float(lon[i]), 4)), []).append(int(i))
    uniq = list(keys)
    batches = [[(j, uniq[j][1], uniq[j][0]) for j in range(a, min(len(uniq), a + size))] for a in range(0, len(uniq), size)]
    results: dict = {}
    failed: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_process_batch, ee, layers, basins, zone, b, names, hist_names): b for b in batches}
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
    cols = {c: [None] * len(out) for c in list(COLUMNS.values()) + list(RAW.values()) + list(ZONE.values())}
    status = ["ok"] * len(out)
    n_sea = 0
    zones_seen = set()
    for j, (key, idxs) in enumerate(keys.items()):
        if j in failed:
            for i in idxs:
                status[i] = "échec Earth Engine : " + failed[j]
            continue
        v, t, zid, zkm2 = results.get(j, ({}, {}, -1, 0.0))
        vv = dict(v)
        up = _num(vv.get("upa"))
        vv["lupa"] = math.log10(max(up, 1e-6)) if up is not None else None
        if _num(vv.get("upa")) is None or _num(vv.get("slope")) is None or (zone["kind"] == "basin" and zid < 0):
            n_sea += len(idxs)
            for i in idxs:
                status[i] = "hors terre ou hors couverture (MERIT Hydro : 90°N à 60°S, terres émergées)"
            continue
        zones_seen.add(zid)
        sc = score_point(vv, t, soil, accum, area)
        t_area = AREA_FIXED_KM2 if area == "fixed" else (10 ** t["lupa"] if "lupa" in t and math.isfinite(t["lupa"]) else None)
        fill = {**{COLUMNS[k]: sc[k] for k in COLUMNS}, **{RAW[k]: _num(vv.get(k)) for k in RAW},
                ZONE["id"]: zid, ZONE["km2"]: round(float(zkm2), 2), ZONE["t_slope"]: _fin(t.get("slope")), ZONE["t_twi"]: _fin(t.get("twi")),
                ZONE["t_area"]: t_area, ZONE["t_curv"]: _fin(t.get("curv"))}
        for c, val in fill.items():
            for i in idxs:
                cols[c][i] = val
    for i in range(len(out)):
        if not valid[i]:
            status[i] = "coordonnées invalides"
    for c, vals in cols.items():
        out[c] = vals
    out[STATUS] = status
    classes = pd.to_numeric(out[COLUMNS["classe"]], errors="coerce")
    summary = {"points": len(out), "invalides": int((~valid).sum()), "pixels_uniques": len(uniq), "lots": len(batches), "echecs": len(failed),
               "hors_terre": n_sea, "facteurs": names, "jeux_absents": [k for k, v in use.items() if not v], "zones": len(zones_seen),
               "zone": dict(zone), "sol": soil, "accumulation": accum, "aire": area,
               "repartition": {LEVELS[int(k)]: int(v) for k, v in classes.value_counts().sort_index().items()}}
    return out, summary


def _fin(x) -> Optional[float]:
    """Seuil présentable : ``None`` si la zone n'a pas de seuil (+inf)."""
    f = _num(x)
    return None if f is None else round(f, 4)


def diagnose(ee_project: Optional[str], ee_account, zone: Optional[dict] = None, soil: str = "cn", accum: str = "classic",
             area: str = "zone") -> list[dict]:
    """Tableau de contrôle : secrets, authentification, jeux de données, bassins, puis notation sur des lieux connus."""
    zone = zone or {"kind": "basin", "level": DEFAULT_LEVEL}
    rows = []
    try:
        ch.init_ee(ee_project, ee_account)
        rows.append({"étape": "authentification", "résultat": "OK", "détail": "Earth Engine initialisé"})
    except TileError as e:
        return [{"étape": "authentification", "résultat": "ÉCHEC", "détail": str(e)}]
    ee = ch._ee()
    try:
        for d in (_LAYER_OK, _ASSET_KIND):
            d.clear()
        use = layers_available(ee, {"worldcover", "gcn", "hsg", "ksat", "geom"})
        rows.append({"étape": "MERIT Hydro", "résultat": "OK", "détail": f"{MERIT_ASSET} : bandes elv et upa présentes"})
        for key, label in (("worldcover", f"ESA WorldCover ({WORLDCOVER_ASSET})"), ("gcn", f"GCN250 ({GCN_ASSET})"),
                           ("hsg", f"HiHydroSoil, groupe hydrologique ({HSG_ASSET})"), ("ksat", f"HiHydroSoil, Ks ({KSAT_ASSET})"),
                           ("geom", f"Geomorpho90m, géomorphons ({GEOM_ASSET})")):
            kind = _ASSET_KIND.get({"hsg": HSG_ASSET, "ksat": KSAT_ASSET, "geom": GEOM_ASSET}.get(key, ""), "")
            rows.append({"étape": label, "résultat": "OK" if use.get(key) else "ABSENT",
                         "détail": ("lisible" + (f" ({kind})" if kind else "")) if use.get(key) else "jeu illisible : option correspondante indisponible"})
        if zone["kind"] == "basin":
            try:
                props = ee.FeatureCollection(BASINS_ASSET.format(level=zone["level"])).first().propertyNames().getInfo()
                rows.append({"étape": f"HydroBASINS niveau {zone['level']}", "résultat": "OK" if "HYBAS_ID" in props else "ÉCHEC",
                             "détail": f"propriétés : {', '.join(props[:8])}"})
            except Exception as e:                              # noqa: BLE001
                rows.append({"étape": f"HydroBASINS niveau {zone['level']}", "résultat": "ÉCHEC", "détail": f"{type(e).__name__}: {str(e)[:160]}"})
        places = {"Vallon des Combes, Biot (France)": (7.10, 43.62), "Alpes, Chamonix": (6.87, 45.92), "Paris": (2.35, 48.85),
                  "Delta du Gange": (90.4, 22.3), "Atlantique (mer)": (-30.0, 40.0)}
        df = pd.DataFrame({"lon": [c[0] for c in places.values()], "lat": [c[1] for c in places.values()]})
        res, s = estimate(df, "lat", "lon", zone, ee_project, ee_account, soil=soil, accum=accum, area=area, workers=1)
        for name, (_, r) in zip(places, res.iterrows()):
            if r[STATUS].startswith("hors terre") and "mer" in name:
                rows.append({"étape": name, "résultat": "OK", "détail": "attendu : aucune valeur en mer (" + r[STATUS] + ")"})
            elif r[STATUS] != "ok":
                rows.append({"étape": name, "résultat": "À CONTRÔLER", "détail": r[STATUS]})
            else:
                raw = " ; ".join(f"{k} {r[RAW[k]]:.3g}" for k in ("cn", "hsg", "ksat", "geom") if r[RAW[k]] is not None and not pd.isna(r[RAW[k]]))
                rows.append({"étape": name, "résultat": "OK", "détail":
                             f"zone {r[ZONE['id']]} ({r[ZONE['km2']]} km²) ; pente {r[RAW['slope']]:.1f}° (seuil {r[ZONE['t_slope']]}) ; aire drainée "
                             f"{r[RAW['upa']]:.2f} km² ; {raw} ; production {r[COLUMNS['prod']]:.2f} ; transfert {r[COLUMNS['transfert']]:.2f} ; "
                             f"accumulation {r[COLUMNS['accum']]:.2f} ; globale {r[COLUMNS['globale']]:.2f} (classe {int(r[COLUMNS['classe']])})"})
    except TileError as e:
        rows.append({"étape": "lecture", "résultat": "ÉCHEC", "détail": str(e)})
    except Exception as e:                                      # noqa: BLE001
        rows.append({"étape": "lecture", "résultat": "ÉCHEC", "détail": f"{type(e).__name__}: {str(e)[:220]}"})
    return rows
