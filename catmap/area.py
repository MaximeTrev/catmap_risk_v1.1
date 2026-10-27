"""Recherche des établissements situés dans un périmètre circulaire (centre + rayon).

Même chaîne que la recherche par nom : requête Overpass -> géométrie et surface -> pays -> classification
NAF (section et division). Différences :
- la sélection se fait par zone (``around``) et non par nom ; on retient les objets portant un tag
  ``shop``, ``amenity`` ou ``brand`` ;
- nodes et ways par défaut, relations multipolygones en option ;
- un rayon maximal (``Settings.area_max_radius_m``) et un nombre de sites maximal protègent contre les réponses
  trop lourdes.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from . import geo
from .config import Settings
from .countries import CountryIndex
from .models import QueryReport
from .naf import NafClassifier
from .overpass import (OverpassClient, OverpassPartialResult, OverpassQueryError, OverpassUnavailable)
from .parsing import CLASS_TAGS, EXTRA_TAGS, ParseStats, _element_geometry, dedupe_sites
from .pipeline import SITE_COLUMNS, GeolocationResult, empty_sites

log = logging.getLogger("catmap.area")

AREA_KEYS = ("shop", "amenity", "brand")
MIN_RADIUS_M = 10.0


@dataclass(frozen=True)
class AreaQuery:
    lat: float
    lon: float
    radius_m: float

    @property
    def label(self) -> str:
        return f"Périmètre {self.lat:.5f}, {self.lon:.5f} - {self.radius_m:.0f} m"

    def validate(self, max_radius_m: float) -> None:
        vals = (self.lat, self.lon, self.radius_m)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError("Périmètre invalide : coordonnées ou rayon non numériques.")
        if not (-90 <= self.lat <= 90 and -180 <= self.lon <= 180):
            raise ValueError("Périmètre invalide : latitude ou longitude hors limites.")
        if self.radius_m < MIN_RADIUS_M:
            raise ValueError(f"Rayon trop petit : minimum {MIN_RADIUS_M:.0f} m.")
        if self.radius_m > max_radius_m:
            raise ValueError(f"Rayon trop grand : maximum {max_radius_m:.0f} m (réglage CATMAP_AREA_MAX_RADIUS_M).")

    @staticmethod
    def from_value(value: dict) -> "AreaQuery":
        try:
            return AreaQuery(float(value["lat"]), float(value["lon"]), float(value["radius_m"]))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Périmètre invalide : {e}") from e


@dataclass(frozen=True)
class AreaOptions:
    include_relations: bool = False     # relations multipolygones (grands sites), plus lourd
    named_only: bool = True             # exige un name ou un brand (écarte bancs, poubelles, etc.)
    dedup_radius_m: float = 25.0        # fusion node/way d'un même site ; 0 = désactivé

    def cache_signature(self) -> dict:
        return {"mode": "perimetre", "rel": self.include_relations, "named": self.named_only,
                "dedup": self.dedup_radius_m}


def build_area_query(area: AreaQuery, options: AreaOptions, settings: Settings) -> str:
    area.validate(settings.area_max_radius_m)
    around = f"(around:{int(round(area.radius_m))},{area.lat:.6f},{area.lon:.6f})"
    head = f"[out:json][timeout:{int(settings.area_timeout_s)}]"
    if settings.maxsize_mb > 0:
        head += f"[maxsize:{int(settings.maxsize_mb) * 1024 * 1024}]"
    elements = ["node", "way"] + (["rel[type=multipolygon]"] if options.include_relations else [])
    lines = [head + ";", "("]
    for el in elements:
        for key in AREA_KEYS:
            lines.append(f"  {el}{around}[{key}];")
    lines += [");", "out body geom;"]
    return "\n".join(lines)


def parse_area_elements(data: dict, area: AreaQuery, options: AreaOptions, max_sites: int) -> tuple[list[dict], ParseStats, int]:
    """Enregistrements « sites » d'une réponse Overpass ; renvoie aussi le nombre de sites hors du cercle."""
    st = ParseStats()
    n_outside = 0
    records: list[dict] = []
    for el in data.get("elements", []):
        st.n_elements += 1
        tags = el.get("tags") or {}
        if not tags or "boundary" in tags:
            st.n_rejected_tags += 1
            continue
        name = (tags.get("name") or "").strip()
        if name.lower() == "unknown":
            name = ""
        brand = (tags.get("brand") or "").strip()
        display = name or brand
        if not display and options.named_only:
            st.n_rejected_name += 1
            continue
        g = _element_geometry(el)
        if g is None:
            st.n_no_geometry += 1
            continue
        dist = geo.haversine_m(area.lon, area.lat, g["lon"], g["lat"])
        if dist > area.radius_m:
            n_outside += 1      # contour qui touche le cercle mais dont le point représentatif est dehors
            continue
        st.n_incomplete_relations += 1 if g["incomplete"] else 0
        rec = {
            "nom_demande": area.label,
            "osm_type": el["type"],
            "osm_id": int(el["id"]),
            "osm_url": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
            "name": display or "(sans nom)",
            "lat": round(g["lat"], 7),
            "long": round(g["lon"], 7),
            "distance_centre_m": round(dist, 1),
            "geom_kind": g["kind"],
            "surface_m2": None if g["area"] is None else round(g["area"], 1),
            "surface_note": g["note"],
            "match_field": "périmètre",
            "match_score": 1.0,
            "match_exact": True,
            "n_fusionnes": 0,
            "_outer": g["_outer"],
        }
        for k in CLASS_TAGS:
            rec[k] = tags.get(k) or None
        for k, col in EXTRA_TAGS.items():
            rec[col] = tags.get(k) or None
        records.append(rec)

    if options.dedup_radius_m and options.dedup_radius_m > 0:
        records, merged = dedupe_sites(records, options.dedup_radius_m)
        st.n_merged = merged
    for r in records:
        r.pop("_outer", None)
    if len(records) > max_sites:
        records.sort(key=lambda r: r["distance_centre_m"])
        records = records[:max_sites]
        st.truncated = True
    return records, st, n_outside


def locate_area(
    client: OverpassClient,
    settings: Settings,
    area: AreaQuery,
    options: AreaOptions,
    country_index: Optional[CountryIndex] = None,
    naf: Optional[NafClassifier] = None,
    snap_km: float = 50.0,
    on_event: Optional[Callable[[str], None]] = None,
) -> GeolocationResult:
    """Ne lève jamais : le statut du rapport porte l'erreur (comme ``search_one``)."""
    t0 = time.monotonic()
    rep = QueryReport(nom_demande=area.label, nom_recherche=area.label)
    records: list[dict] = []
    try:
        query = build_area_query(area, options, settings)
        print(f"[CatMap] Requête Overpass pour « {area.label} » :\n{query}", flush=True)
        try:
            fr = client.fetch(query, on_event=on_event)
            data, rep.endpoint, rep.snapshot, rep.depuis_cache = fr.data, fr.endpoint, fr.snapshot, fr.from_cache
        except OverpassPartialResult as p:
            data, rep.endpoint = p.data, p.endpoint
            rep.snapshot = str((p.data.get("osm3s") or {}).get("timestamp_osm_base") or "")
            rep.statut = "partiel"
            rep.message = f"Résultat partiel (non mis en cache) : {p.remark}"
        records, s, n_outside = parse_area_elements(data, area, options, settings.area_max_sites)
        rep.n_elements_osm = s.n_elements
        rep.n_rejetes_nom = s.n_rejected_name + s.n_rejected_tags
        rep.n_sans_geometrie, rep.n_doublons_fusionnes, rep.n_sites = s.n_no_geometry, s.n_merged, len(records)
        notes = []
        if s.truncated:
            notes.append(f"limité aux {settings.area_max_sites} sites les plus proches du centre : réduisez le rayon")
        if n_outside:
            notes.append(f"{n_outside} objet(s) à cheval sur le cercle écarté(s) (point représentatif hors périmètre)")
        if s.n_incomplete_relations:
            notes.append(f"{s.n_incomplete_relations} relation(s) incomplète(s)")
        if notes:
            rep.message = (rep.message + " ; " if rep.message else "") + ", ".join(notes)
        if rep.statut != "partiel":
            rep.statut = "ok" if records else "aucun résultat"
    except OverpassQueryError as e:
        rep.statut, rep.message = "erreur", f"Requête refusée : {e}"
    except OverpassUnavailable as e:
        rep.statut, rep.message = "erreur", f"Serveurs Overpass indisponibles : {e}"
    except ValueError as e:
        rep.statut, rep.message = "erreur", str(e)
    except Exception as e:  # noqa: BLE001
        log.exception("Erreur inattendue pour le périmètre %s", area.label)
        rep.statut, rep.message = "erreur", f"{type(e).__name__}: {e}"
    rep.duree_s = round(time.monotonic() - t0, 2)
    print(f"[CatMap] « {area.label} » : {rep.duree_s:.1f} s - statut : {rep.statut}, {rep.n_sites} site(s)"
          + (" (cache)" if rep.depuis_cache else "") + (f" - {rep.endpoint}" if rep.endpoint else ""), flush=True)

    for r in records:
        r["snapshot_osm"] = rep.snapshot
    if records:
        sites = pd.DataFrame(records)
        for c in SITE_COLUMNS:
            if c not in sites.columns:
                sites[c] = None
        if country_index is not None:
            geo_df = country_index.locate(sites["long"].to_numpy(float), sites["lat"].to_numpy(float), snap_km=snap_km)
            for c in geo_df.columns:
                sites[c] = geo_df[c].to_numpy()
        if naf is not None:
            sites = naf.classify(sites)
        order = ["name"] + [c for c in SITE_COLUMNS if c != "name"]
        i = order.index("long") + 1
        order.insert(i, "distance_centre_m")
        sites = sites[order + [c for c in sites.columns if c not in order]]
    else:
        sites = empty_sites()
        sites["distance_centre_m"] = pd.Series(dtype="float")
    report_df = pd.DataFrame([rep.as_dict()])
    meta = {
        "n_requetes": 1,
        "n_sites": int(len(sites)),
        "snapshots": [rep.snapshot] if rep.snapshot else [],
        "options": options.cache_signature(),
        "endpoints": [rep.endpoint] if rep.endpoint else [],
        "area": {"lat": area.lat, "lon": area.lon, "radius_m": area.radius_m},
    }
    return GeolocationResult(sites=sites, report=report_df, meta=meta)
