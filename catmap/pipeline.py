"""Orchestration de la géolocalisation.

Sans dépendance à Streamlit (corrige A10 : le décorateur de chronométrage dépendant de
``st.session_state`` disparaît ; les durées sont mesurées dans le rapport).

- un rapport par entreprise **demandée** (corrige A6 : « sans résultat » fiable) ;
- un échec sur une entreprise n'arrête jamais le lot ;
- DataFrame vide mais correctement typé quand rien n'est trouvé (plus de KeyError) ;
- parallélisme borné, réglable (utile avec une instance Overpass locale).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import pandas as pd

from .config import Settings
from .countries import CountryIndex
from .models import NameQuery, QueryReport, SearchOptions
from .naf import NafClassifier
from .overpass import (OverpassClient, OverpassPartialResult, OverpassQueryError,
                       OverpassUnavailable, build_query)
from .parsing import parse_elements

log = logging.getLogger("catmap.pipeline")

SITE_COLUMNS = [
    "nom_demande", "name", "osm_type", "osm_id", "osm_url", "lat", "long", "geom_kind",
    "surface_m2", "surface_note", "match_field", "match_score", "match_exact", "n_fusionnes",
    "pays", "pays_iso3", "pays_iso2", "pays_methode", "pays_dist_km",
    "naf_section_code", "naf_section", "naf_division_code", "naf_division", "naf_source", "naf_confidence",
    "brand", "operator", "brand_wikidata", "shop", "amenity", "office", "craft", "industrial", "man_made",
    "power", "healthcare", "tourism", "leisure", "landuse", "building",
    "addr_street", "addr_housenumber", "addr_postcode", "addr_city", "addr_country", "website",
    "snapshot_osm",
]


def empty_sites() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SITE_COLUMNS})


@dataclass
class GeolocationResult:
    sites: pd.DataFrame
    report: pd.DataFrame
    meta: dict = field(default_factory=dict)


def search_one(client: OverpassClient, settings: Settings, q: NameQuery,
               options: SearchOptions) -> tuple[list[dict], QueryReport]:
    """Recherche d'UNE entreprise ; ne lève jamais (le statut porte l'erreur)."""
    t0 = time.monotonic()
    rep = QueryReport(nom_demande=q.raw, nom_recherche=q.search_name, wikidata=q.wikidata)
    records: list[dict] = []
    try:
        query = build_query(q, options, settings)
        try:
            fr = client.fetch(query)
            data, rep.endpoint, rep.snapshot, rep.depuis_cache = fr.data, fr.endpoint, fr.snapshot, fr.from_cache
        except OverpassPartialResult as p:
            data, rep.endpoint = p.data, p.endpoint
            rep.snapshot = str((p.data.get("osm3s") or {}).get("timestamp_osm_base") or "")
            rep.statut = "partiel"
            rep.message = f"Résultat partiel (non mis en cache) : {p.remark}"
        parsed = parse_elements(data, q, options, max_sites=settings.max_sites_per_company)
        records = parsed.records
        s = parsed.stats
        rep.n_elements_osm, rep.n_rejetes_nom = s.n_elements, s.n_rejected_name + s.n_rejected_score
        rep.n_sans_geometrie, rep.n_doublons_fusionnes, rep.n_sites = s.n_no_geometry, s.n_merged, len(records)
        notes = []
        if s.truncated:
            notes.append(f"tronqué à {settings.max_sites_per_company} sites")
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
    except Exception as e:  # noqa: BLE001 - un échec ne doit pas interrompre le lot
        log.exception("Erreur inattendue pour %s", q.raw)
        rep.statut, rep.message = "erreur", f"{type(e).__name__}: {e}"
    rep.duree_s = round(time.monotonic() - t0, 2)
    return records, rep


def geolocate(
    queries: Sequence[NameQuery],
    options: SearchOptions,
    client: OverpassClient,
    settings: Settings,
    country_index: Optional[CountryIndex] = None,
    naf: Optional[NafClassifier] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    snap_km: float = 50.0,
) -> GeolocationResult:
    uniq: dict[tuple, NameQuery] = {}
    for q in queries:
        uniq.setdefault((q.raw.casefold(), q.wikidata), q)
    todo = list(uniq.values())
    total = len(todo)
    results: dict[int, tuple[list[dict], QueryReport]] = {}

    workers = max(1, min(settings.max_workers, total or 1))
    if workers == 1:
        for i, q in enumerate(todo):
            results[i] = search_one(client, settings, q, options)
            if progress:
                progress(i + 1, total, q.raw)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(search_one, client, settings, q, options): i for i, q in enumerate(todo)}
            done = 0
            for f in as_completed(futs):
                results[futs[f]] = f.result()
                done += 1
                if progress:
                    progress(done, total, todo[futs[f]].raw)

    all_records: list[dict] = []
    reports: list[dict] = []
    for i in range(total):
        recs, rep = results[i]
        for r in recs:
            r["snapshot_osm"] = rep.snapshot
        all_records.extend(recs)
        reports.append(rep.as_dict())
    report_df = pd.DataFrame(reports)

    if all_records:
        sites = pd.DataFrame(all_records)
        for c in SITE_COLUMNS:
            if c not in sites.columns:
                sites[c] = None
        if country_index is not None:
            geo_df = country_index.locate(sites["long"].to_numpy(float), sites["lat"].to_numpy(float), snap_km=snap_km)
            for c in geo_df.columns:
                sites[c] = geo_df[c].to_numpy()
        if naf is not None:
            sites = naf.classify(sites)
        sites = sites[SITE_COLUMNS + [c for c in sites.columns if c not in SITE_COLUMNS]]
    else:
        sites = empty_sites()

    meta = {
        "n_requetes": total,
        "n_sites": int(len(sites)),
        "snapshots": sorted({s for s in report_df.get("snapshot", []) if s}),
        "options": options.cache_signature(),
        "endpoints": sorted({e for e in report_df.get("endpoint", []) if e}),
    }
    return GeolocationResult(sites=sites, report=report_df, meta=meta)
