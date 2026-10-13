"""Géométrie sans dépendance externe (ni shapely, ni pyproj).

Pourquoi du Python pur ?
- testable partout (aucune roue binaire à installer) ;
- surface **exacte sur l'ellipsoïde WGS84** sans choisir de fuseau UTM
  (corrige A8 : plus d'erreur près des limites de fuseau, pas de Transformer à recréer) ;
- reconstruction des multipolygones OSM (relations) : lève l'hypothèse H5.

Convention : les coordonnées sont des tuples ``(lon, lat)`` en degrés WGS84.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional, Sequence

Coord = tuple[float, float]
Ring = list[Coord]
Polygon = tuple[Ring, list[Ring]]  # (anneau extérieur, [trous])

# --------------------------------------------------------------------------
# Constantes WGS84 et surface authalique
# --------------------------------------------------------------------------
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_E = math.sqrt(_E2)


def _q(sin_phi: float) -> float:
    return (1.0 - _E2) * (
        sin_phi / (1.0 - _E2 * sin_phi * sin_phi)
        - (1.0 / (2.0 * _E)) * math.log((1.0 - _E * sin_phi) / (1.0 + _E * sin_phi))
    )


_QP = _q(1.0)
_RQ2 = (_A * _A) * _QP / 2.0  # rayon authalique au carré


def _sin_beta(lat_deg: float) -> float:
    """Sinus de la latitude authalique (projection cylindrique équivalente de Lambert)."""
    return _q(math.sin(math.radians(lat_deg))) / _QP


def ring_area_m2(ring: Sequence[Coord]) -> float:
    """Surface (m²) d'un anneau, en valeur absolue.

    Les arêtes sont des segments droits dans l'espace (longitude, sin latitude authalique) :
    surface exacte sur l'ellipsoïde pour ce tracé ; l'écart avec des arêtes géodésiques est
    négligeable pour des bâtiments et sites industriels (arêtes de quelques km au plus).
    """
    if len(ring) < 3:
        return 0.0
    pts = list(ring)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    total = 0.0
    prev_lon, prev_lat = pts[0]
    prev_s = _sin_beta(prev_lat)
    for lon, lat in pts[1:]:
        s = _sin_beta(lat)
        dlon = lon - prev_lon
        if dlon > 180.0:
            dlon -= 360.0
        elif dlon < -180.0:
            dlon += 360.0
        total += math.radians(dlon) * (prev_s + s)
        prev_lon, prev_s = lon, s
    return abs(total) * _RQ2 / 2.0


def polygon_area_m2(poly: Polygon) -> float:
    outer, holes = poly
    area = ring_area_m2(outer) - sum(ring_area_m2(h) for h in holes)
    return max(area, 0.0)


# --------------------------------------------------------------------------
# Prédicats et points caractéristiques
# --------------------------------------------------------------------------
def point_in_ring(x: float, y: float, ring: Sequence[Coord]) -> bool:
    """Test point dans anneau (règle pair-impair)."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


def point_in_polygon(x: float, y: float, poly: Polygon) -> bool:
    outer, holes = poly
    return point_in_ring(x, y, outer) and not any(point_in_ring(x, y, h) for h in holes)


def ring_centroid(ring: Sequence[Coord]) -> Coord:
    """Centroïde planaire corrigé de cos(latitude) (suffisant pour un site)."""
    pts = list(ring)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not pts:
        raise ValueError("anneau vide")
    lat0 = sum(p[1] for p in pts) / len(pts)
    k = max(math.cos(math.radians(lat0)), 1e-6)
    a = cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i][0] * k, pts[i][1]
        x1, y1 = pts[(i + 1) % n][0] * k, pts[(i + 1) % n][1]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-18:
        return (sum(p[0] for p in pts) / n, lat0)
    a *= 0.5
    return (cx / (6.0 * a) / k, cy / (6.0 * a))


def representative_point(poly: Polygon) -> Coord:
    """Point garanti dans le polygone : centroïde s'il est dedans, sinon milieu du plus
    large intervalle d'une droite horizontale (corrige H7 : centre hors du bâti en L)."""
    outer, holes = poly
    c = ring_centroid(outer)
    if point_in_polygon(c[0], c[1], poly):
        return c
    ys = [p[1] for p in outer]
    y0 = c[1] if min(ys) <= c[1] <= max(ys) else (min(ys) + max(ys)) / 2.0
    for y in (y0, (min(ys) + max(ys)) / 2.0, min(ys) + 0.25 * (max(ys) - min(ys)),
              min(ys) + 0.75 * (max(ys) - min(ys))):
        xs: list[float] = []
        for ring in [outer, *holes]:
            n = len(ring)
            for i in range(n):
                xa, ya = ring[i]
                xb, yb = ring[(i + 1) % n]
                if (ya > y) != (yb > y):
                    xs.append(xa + (y - ya) * (xb - xa) / (yb - ya))
        xs.sort()
        best: Optional[tuple[float, float]] = None
        for i in range(0, len(xs) - 1, 2):
            w = xs[i + 1] - xs[i]
            if best is None or w > best[0]:
                best = (w, (xs[i] + xs[i + 1]) / 2.0)
        if best is not None and best[0] > 0:
            cand = (best[1], y)
            if point_in_polygon(cand[0], cand[1], poly):
                return cand
    return outer[0]


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


# --------------------------------------------------------------------------
# Assemblage des multipolygones OSM
# --------------------------------------------------------------------------
def _key(p: Coord) -> Coord:
    return (round(p[0], 7), round(p[1], 7))


def assemble_rings(segments: Iterable[Sequence[Coord]]) -> tuple[list[Ring], list[Ring]]:
    """Chaîne des segments (ways membres) en anneaux fermés.

    Retourne ``(anneaux_fermés, chaînes_ouvertes)`` ; une chaîne ouverte signale une
    relation incomplète (membre absent de l'extrait).
    """
    segs = [list(map(tuple, s)) for s in segments if len(s) >= 2]
    ends: dict[Coord, list[int]] = defaultdict(list)
    for i, s in enumerate(segs):
        ends[_key(s[0])].append(i)
        ends[_key(s[-1])].append(i)
    used = [False] * len(segs)
    rings: list[Ring] = []
    open_chains: list[Ring] = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        chain: Ring = list(segs[i])
        while _key(chain[0]) != _key(chain[-1]):
            nxt = None
            k = _key(chain[-1])
            for j in ends[k]:
                if not used[j]:
                    nxt = j
                    break
            if nxt is not None:
                seg = segs[nxt]
                if _key(seg[0]) != k:
                    seg = seg[::-1]
                chain.extend(seg[1:])
                used[nxt] = True
                continue
            k = _key(chain[0])
            for j in ends[k]:
                if not used[j]:
                    nxt = j
                    break
            if nxt is None:
                break
            seg = segs[nxt]
            if _key(seg[-1]) != k:
                seg = seg[::-1]
            chain = seg[:-1] + chain
            used[nxt] = True
        if len(chain) >= 4 and _key(chain[0]) == _key(chain[-1]):
            rings.append(chain)
        else:
            open_chains.append(chain)
    return rings, open_chains


def build_multipolygon(members: Iterable[dict]) -> tuple[list[Polygon], int]:
    """Construit les polygones d'une relation ``type=multipolygon`` (sortie ``out geom``).

    Retourne ``(polygones, nb_chaînes_ouvertes)``. Les trous sont rattachés à l'anneau
    extérieur qui les contient.
    """
    outers: list[Ring] = []
    inners: list[Ring] = []
    for m in members:
        if m.get("type") != "way":
            continue
        geom = m.get("geometry")
        if not geom:
            continue
        seg = [(p["lon"], p["lat"]) for p in geom if p]
        (inners if m.get("role") == "inner" else outers).append(seg)
    outer_rings, open_o = assemble_rings(outers)
    inner_rings, open_i = assemble_rings(inners)
    polys: list[list] = [[o, []] for o in outer_rings]
    for h in inner_rings:
        px, py = h[0]
        for p in polys:
            if point_in_ring(px, py, p[0]):
                p[1].append(h)
                break
    return [(o, hs) for o, hs in polys], len(open_o)
