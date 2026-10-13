"""Transformation d'une réponse Overpass JSON en enregistrements « sites ».

Corrige, par rapport à la v1 :
- A8/H8 : surface uniquement pour les contours **fermés** (un way ouvert n'a plus de surface
          fictive) ; surface géodésique exacte ; multipolygones assemblés (trous compris) ;
          surface inconnue = ``None`` (et non 0, indistinguable d'un point) ;
- A9    : plus de ``hasattr(way, "center_lat")`` ; la géométrie est lue explicitement ;
- H7    : point représentatif garanti dans le polygone (bâtiments en L / U) ;
- A16   : identifiant OSM et URL conservés, tags ``brand``, ``operator``, ``brand:wikidata``,
          adresse ; fusion des doublons node/way d'un même site ; score de correspondance.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Optional

from . import geo
from .models import NameQuery, SearchOptions
from .names import MatchInfo, NameMatcher, normalize_name

# Tags conservés (None si absents : plus de chaîne "N/A" qui échappe à isna(), cf. extract_tags v1)
CLASS_TAGS = ("shop", "amenity", "office", "craft", "industrial", "man_made", "power",
              "healthcare", "tourism", "leisure", "landuse", "building")
EXTRA_TAGS = {
    "brand": "brand", "operator": "operator", "brand:wikidata": "brand_wikidata",
    "addr:street": "addr_street", "addr:housenumber": "addr_housenumber",
    "addr:postcode": "addr_postcode", "addr:city": "addr_city", "addr:country": "addr_country",
    "website": "website",
}
_KIND_RANK = {"polygon": 0, "multipolygon": 0, "line": 1, "point": 2}


@dataclass
class ParseStats:
    n_elements: int = 0
    n_rejected_name: int = 0
    n_rejected_score: int = 0
    n_no_geometry: int = 0
    n_incomplete_relations: int = 0
    n_merged: int = 0
    truncated: bool = False


@dataclass
class ParseResult:
    records: list[dict] = field(default_factory=list)
    stats: ParseStats = field(default_factory=ParseStats)


# --------------------------------------------------------------------------
# Géométrie d'un élément
# --------------------------------------------------------------------------
def _element_geometry(el: dict) -> Optional[dict]:
    """Retourne ``{lat, lon, kind, area, note, _outer}`` ou None si non localisable."""
    t = el.get("type")
    if t == "node":
        if "lat" in el and "lon" in el:
            return {"lat": float(el["lat"]), "lon": float(el["lon"]), "kind": "point",
                    "area": None, "note": "", "_outer": None, "incomplete": 0}
        return None

    if t == "way":
        pts = [(float(p["lon"]), float(p["lat"])) for p in (el.get("geometry") or []) if p]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            poly = (pts, [])
            lon, lat = geo.representative_point(poly)
            return {"lat": lat, "lon": lon, "kind": "polygon", "area": geo.polygon_area_m2(poly),
                    "note": "", "_outer": pts, "incomplete": 0}
        if pts:  # way ouvert : pas de surface (v1 : surface fictive après fermeture forcée)
            b = el.get("bounds")
            if b:
                lat, lon = (b["minlat"] + b["maxlat"]) / 2.0, (b["minlon"] + b["maxlon"]) / 2.0
            else:
                lat = sum(p[1] for p in pts) / len(pts)
                lon = sum(p[0] for p in pts) / len(pts)
            return {"lat": lat, "lon": lon, "kind": "line", "area": None,
                    "note": "contour non fermé : surface non calculée", "_outer": None, "incomplete": 0}
        c = el.get("center")
        if c:
            return {"lat": float(c["lat"]), "lon": float(c["lon"]), "kind": "line", "area": None,
                    "note": "géométrie absente : centre seul", "_outer": None, "incomplete": 0}
        return None

    if t == "relation":
        polys, incomplete = geo.build_multipolygon(el.get("members") or [])
        if not polys:
            b = el.get("bounds")
            if b:
                return {"lat": (b["minlat"] + b["maxlat"]) / 2.0, "lon": (b["minlon"] + b["maxlon"]) / 2.0,
                        "kind": "line", "area": None, "note": "relation sans contour exploitable",
                        "_outer": None, "incomplete": incomplete}
            return None
        largest = max(polys, key=lambda p: geo.ring_area_m2(p[0]))
        lon, lat = geo.representative_point(largest)
        area = sum(geo.polygon_area_m2(p) for p in polys)
        note = "relation incomplète : surface sous-estimée" if incomplete else ""
        return {"lat": lat, "lon": lon, "kind": "multipolygon", "area": area, "note": note,
                "_outer": largest[0], "incomplete": incomplete}
    return None


# --------------------------------------------------------------------------
# Analyse d'une réponse
# --------------------------------------------------------------------------
def parse_elements(data: dict, query: NameQuery, options: SearchOptions,
                   max_sites: int = 50000) -> ParseResult:
    matcher = NameMatcher(query.search_name, query.wikidata)
    res = ParseResult()
    st = res.stats
    for el in data.get("elements", []):
        st.n_elements += 1
        tags: dict[str, Any] = el.get("tags") or {}
        if not tags:
            continue
        name = (tags.get("name") or "").strip()
        if name.lower() == "unknown":
            name = ""
        display = name or (tags.get("brand") or "").strip()
        if not display:
            continue

        info = matcher.match_tags(tags, include_operator=options.include_operator)
        if info is None:
            if options.strict_word_match:
                st.n_rejected_name += 1
                continue
            info = MatchInfo("regex", 0.30, False)
        if info.score < options.min_score:
            st.n_rejected_score += 1
            continue

        g = _element_geometry(el)
        if g is None:
            st.n_no_geometry += 1
            continue
        st.n_incomplete_relations += 1 if g["incomplete"] else 0

        rec: dict[str, Any] = {
            "nom_demande": query.raw,
            "osm_type": el["type"],
            "osm_id": int(el["id"]),
            "osm_url": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
            "name": display,
            "lat": round(g["lat"], 7),
            "long": round(g["lon"], 7),
            "geom_kind": g["kind"],
            "surface_m2": None if g["area"] is None else round(g["area"], 1),
            "surface_note": g["note"],
            "match_field": info.field,
            "match_score": info.score,
            "match_exact": info.exact,
            "n_fusionnes": 0,
            "_outer": g["_outer"],
        }
        for k in CLASS_TAGS:
            rec[k] = tags.get(k) or None
        for k, col in EXTRA_TAGS.items():
            rec[col] = tags.get(k) or None
        res.records.append(rec)

    if options.dedup_radius_m and options.dedup_radius_m > 0:
        res.records, merged = dedupe_sites(res.records, options.dedup_radius_m)
        st.n_merged = merged
    for r in res.records:
        r.pop("_outer", None)

    if len(res.records) > max_sites:
        res.records.sort(key=lambda r: (-r["match_score"], _KIND_RANK[r["geom_kind"]]))
        res.records = res.records[:max_sites]
        st.truncated = True
    return res


# --------------------------------------------------------------------------
# Fusion des doublons (un même site cartographié en node + way)
# --------------------------------------------------------------------------
def _priority(r: dict) -> tuple:
    return (_KIND_RANK[r["geom_kind"]], -r["match_score"], -(r["surface_m2"] or 0.0))


def dedupe_sites(records: list[dict], radius_m: float) -> tuple[list[dict], int]:
    """Fusionne les enregistrements de même nom normalisé situés à moins de ``radius_m``
    (ou un point situé dans le contour d'un polygone de même nom).

    Le contour est conservé de préférence au point ; les tags de classification absents du
    conservé sont repris de l'élément fusionné.
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(normalize_name(r["name"]), []).append(r)

    out: list[dict] = []
    merged = 0
    lat_window = radius_m / 111_000.0
    for _, grp in groups.items():
        grp.sort(key=_priority)
        kept: list[dict] = []
        kept_lats: list[tuple[float, int]] = []  # (lat, index dans kept), trié
        for r in grp:
            target = None
            lo = bisect.bisect_left(kept_lats, (r["lat"] - lat_window, -1))
            hi = bisect.bisect_right(kept_lats, (r["lat"] + lat_window, len(kept)))
            for _, idx in kept_lats[lo:hi]:
                k = kept[idx]
                if geo.haversine_m(k["long"], k["lat"], r["long"], r["lat"]) <= radius_m:
                    target = k
                    break
                if k["_outer"] is not None and r["geom_kind"] == "point" \
                        and geo.point_in_ring(r["long"], r["lat"], k["_outer"]):
                    target = k
                    break
            if target is None:
                kept.append(r)
                bisect.insort(kept_lats, (r["lat"], len(kept) - 1))
            else:
                merged += 1
                target["n_fusionnes"] += 1
                for col in (*CLASS_TAGS, *EXTRA_TAGS.values()):
                    if target.get(col) is None and r.get(col) is not None:
                        target[col] = r[col]
        out.extend(kept)
    return out, merged
