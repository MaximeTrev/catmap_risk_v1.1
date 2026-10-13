"""Localisation de points dans un ensemble de polygones (pays, départements…).

Vectorisé avec NumPy (règle pair-impair sur toutes les arêtes d'une entité, donc trous et
multi-parties gérés). Remplace ``geopandas.sjoin`` de la v1 et corrige :
- A7  : plus d'argument mutable ``Pays=[]`` partagé entre appels ;
- A7  : pas de lignes dupliquées (une seule entité retenue par point) ;
- H9  : les points proches d'une côte ou d'une frontière (fichier généralisé) sont rattachés
        à l'entité la plus proche si elle est à moins de ``snap_km`` km, avec la méthode
        indiquée (``within`` ou ``nearest``).
"""
from __future__ import annotations

import numpy as np


class PolygonIndex:
    def __init__(self, shapes_rings: list[list[np.ndarray]]):
        self.n = len(shapes_rings)
        self._edges: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._bbox = np.full((self.n, 4), np.nan)  # minx, miny, maxx, maxy
        for i, rings in enumerate(shapes_rings):
            x1, y1, x2, y2 = [], [], [], []
            allpts = []
            for r in rings:
                r = np.asarray(r, dtype=float)
                if len(r) < 3:
                    continue
                if not np.array_equal(r[0], r[-1]):
                    r = np.vstack([r, r[:1]])
                x1.append(r[:-1, 0]); y1.append(r[:-1, 1])
                x2.append(r[1:, 0]); y2.append(r[1:, 1])
                allpts.append(r)
            if not allpts:
                self._edges.append((np.empty(0),) * 4)  # type: ignore[arg-type]
                continue
            self._edges.append((np.concatenate(x1), np.concatenate(y1), np.concatenate(x2), np.concatenate(y2)))
            pts = np.vstack(allpts)
            self._bbox[i] = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        area = (self._bbox[:, 2] - self._bbox[:, 0]) * (self._bbox[:, 3] - self._bbox[:, 1])
        area = np.nan_to_num(area, nan=np.inf)
        # les petites entités d'abord : un enclave (Monaco, Vatican…) l'emporte sur l'englobant
        self._order = np.argsort(area, kind="stable")

    # ------------------------------------------------------------------
    def _inside(self, px: np.ndarray, py: np.ndarray, si: int) -> np.ndarray:
        x1, y1, x2, y2 = self._edges[si]
        out = np.zeros(len(px), dtype=bool)
        if len(x1) == 0:
            return out
        dy = y2 - y1
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(dy != 0, (x2 - x1) / dy, 0.0)
        step = max(1, 3_000_000 // len(x1))
        for s in range(0, len(px), step):
            X = px[s:s + step, None]
            Y = py[s:s + step, None]
            cond = (y1[None, :] > Y) != (y2[None, :] > Y)
            x_int = x1[None, :] + (Y - y1[None, :]) * slope[None, :]
            out[s:s + step] = ((cond & (X < x_int)).sum(axis=1) & 1).astype(bool)
        return out

    def _min_dist_km(self, px: np.ndarray, py: np.ndarray, si: int) -> np.ndarray:
        x1, y1, x2, y2 = self._edges[si]
        if len(x1) == 0:
            return np.full(len(px), np.inf)
        res = np.empty(len(px))
        step = max(1, 1_000_000 // len(x1))
        for s in range(0, len(px), step):
            X, Y = px[s:s + step], py[s:s + step]
            kx = (111.32 * np.cos(np.radians(Y)))[:, None]
            ky = 110.57
            X1 = (x1[None, :] - X[:, None]) * kx
            Y1 = (y1[None, :] - Y[:, None]) * ky
            X2 = (x2[None, :] - X[:, None]) * kx
            Y2 = (y2[None, :] - Y[:, None]) * ky
            Dx, Dy = X2 - X1, Y2 - Y1
            L2 = Dx * Dx + Dy * Dy
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(L2 > 0, -(X1 * Dx + Y1 * Dy) / L2, 0.0)
            t = np.clip(t, 0.0, 1.0)
            cx, cy = X1 + t * Dx, Y1 + t * Dy
            res[s:s + step] = np.sqrt(cx * cx + cy * cy).min(axis=1)
        return res

    # ------------------------------------------------------------------
    def locate(self, lon, lat, snap_km: float = 0.0):
        """Retourne ``(indices, méthodes, distances_km)`` alignés sur les entrées.

        ``indices`` vaut -1 si aucune entité ; ``méthodes`` ∈ {"within", "nearest", ""}.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        n = len(lon)
        idx_full = np.full(n, -1, dtype=int)
        meth_full = np.full(n, "", dtype=object)
        dist_full = np.full(n, np.nan)
        valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
        if not valid.any():
            return idx_full, meth_full, dist_full

        pts = np.column_stack([np.round(lon[valid], 4), np.round(lat[valid], 4)])
        uniq, inv = np.unique(pts, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        ux, uy = uniq[:, 0], uniq[:, 1]
        res = np.full(len(uniq), -1, dtype=int)
        meth = np.full(len(uniq), "", dtype=object)
        dist = np.full(len(uniq), np.nan)

        for si in self._order:
            minx, miny, maxx, maxy = self._bbox[si]
            if np.isnan(minx):
                continue
            cand = np.where((res < 0) & (ux >= minx) & (ux <= maxx) & (uy >= miny) & (uy <= maxy))[0]
            if len(cand) == 0:
                continue
            ins = self._inside(ux[cand], uy[cand], si)
            hit = cand[ins]
            res[hit] = si
            meth[hit] = "within"
            dist[hit] = 0.0
            if (res >= 0).all():
                break

        if snap_km > 0:
            todo = np.where(res < 0)[0]
            if len(todo):
                best = np.full(len(todo), np.inf)
                best_i = np.full(len(todo), -1, dtype=int)
                tx, ty = ux[todo], uy[todo]
                dlat = snap_km / 110.5
                dlon = snap_km / (111.3 * np.maximum(np.cos(np.radians(ty)), 0.05))
                for si in range(self.n):
                    minx, miny, maxx, maxy = self._bbox[si]
                    if np.isnan(minx):
                        continue
                    sel = np.where((ty >= miny - dlat) & (ty <= maxy + dlat)
                                   & (tx >= minx - dlon) & (tx <= maxx + dlon))[0]
                    if len(sel) == 0:
                        continue
                    d = self._min_dist_km(tx[sel], ty[sel], si)
                    better = d < best[sel]
                    best[sel[better]] = d[better]
                    best_i[sel[better]] = si
                ok = best <= snap_km
                res[todo[ok]] = best_i[ok]
                meth[todo[ok]] = "nearest"
                dist[todo[ok]] = best[ok]

        idx_full[valid] = res[inv]
        meth_full[valid] = meth[inv]
        dist_full[valid] = dist[inv]
        return idx_full, meth_full, dist_full
