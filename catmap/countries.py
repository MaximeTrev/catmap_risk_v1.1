"""Attribution du pays à chaque site (jointure spatiale hors ligne)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .polygon_index import PolygonIndex
from .shapefile import read_polygons

DEFAULT_SHP = DATA_DIR / "country_worldwide" / "curiexplore-pays.shp"


class CountryIndex:
    """Wrapper autour de ``PolygonIndex`` pour le shapefile pays CuriExplore (WGS84)."""

    def __init__(self, shp_path: Optional[Path] = None):
        shapes = read_polygons(shp_path or DEFAULT_SHP)
        self.attrs = [s.attrs for s in shapes]
        self._index = PolygonIndex([s.rings for s in shapes])
        self._names = np.array([a.get("name_fr") or "" for a in self.attrs], dtype=object)
        self._iso3 = np.array([a.get("iso3") or "" for a in self.attrs], dtype=object)
        self._iso2 = np.array([a.get("iso2") or "" for a in self.attrs], dtype=object)

    def locate(self, lon, lat, snap_km: float = 50.0) -> pd.DataFrame:
        """Colonnes : ``pays`` (nom FR), ``pays_iso3``, ``pays_iso2``, ``pays_methode``
        (``within`` / ``nearest``), ``pays_dist_km`` (0 si within). Vide si non trouvé."""
        idx, meth, dist = self._index.locate(lon, lat, snap_km=snap_km)
        found = idx >= 0
        safe = np.where(found, idx, 0)
        return pd.DataFrame({
            "pays": np.where(found, self._names[safe], None),
            "pays_iso3": np.where(found, self._iso3[safe], None),
            "pays_iso2": np.where(found, self._iso2[safe], None),
            "pays_methode": np.where(found, meth, None),
            "pays_dist_km": np.where(found, dist, np.nan),
        })


@lru_cache(maxsize=2)
def get_country_index(path: Optional[str] = None) -> CountryIndex:
    """Charge le shapefile une seule fois par processus."""
    return CountryIndex(Path(path) if path else None)
