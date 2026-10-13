"""Lecteur minimal de shapefile (types Polygon, PolygonZ, PolygonM) sans GDAL/geopandas.

Suffit pour ``curiexplore-pays.shp`` et rend la jointure pays testable et légère à déployer.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

_POLYGON_TYPES = {5, 15, 25}


@dataclass
class Shape:
    rings: list[np.ndarray]   # chaque anneau : tableau (n, 2) de (lon, lat)
    attrs: dict


def _read_dbf(path: Path, encoding: str) -> list[dict]:
    with open(path, "rb") as fh:
        head = fh.read(32)
        n_records = struct.unpack("<I", head[4:8])[0]
        header_len, record_len = struct.unpack("<HH", head[8:12])
        fields = []
        while True:
            d = fh.read(32)
            if not d or d[0] == 0x0D:
                break
            name = d[:11].split(b"\x00")[0].decode("ascii", "ignore")
            fields.append((name, d[11:12].decode("ascii", "ignore"), d[16]))
        fh.seek(header_len)
        rows = []
        for _ in range(n_records):
            rec = fh.read(record_len)
            if len(rec) < record_len:
                break
            pos = 1  # octet d'effacement
            row = {}
            for name, ftype, flen in fields:
                raw = rec[pos:pos + flen]
                pos += flen
                val = raw.decode(encoding, "replace").strip()
                if ftype in ("N", "F"):
                    try:
                        val = float(val) if val else None  # type: ignore[assignment]
                    except ValueError:
                        val = None  # type: ignore[assignment]
                row[name] = val
            rows.append(row)
    return rows


def read_polygons(shp_path: Union[str, Path]) -> list[Shape]:
    """Lit un shapefile de polygones ; ``.dbf`` et ``.cpg`` (encodage) sont lus s'ils existent."""
    shp = Path(shp_path)
    dbf = shp.with_suffix(".dbf")
    cpg = shp.with_suffix(".cpg")
    encoding = "utf-8"
    if cpg.exists():
        enc = cpg.read_text(errors="ignore").strip()
        if enc:
            encoding = {"UTF-8": "utf-8", "UTF8": "utf-8"}.get(enc.upper(), enc)
    attrs = _read_dbf(dbf, encoding) if dbf.exists() else []

    data = shp.read_bytes()
    if struct.unpack(">i", data[0:4])[0] != 9994:
        raise ValueError(f"{shp} n'est pas un shapefile valide")
    shapes: list[Shape] = []
    pos = 100
    idx = 0
    while pos + 8 <= len(data):
        _, content_words = struct.unpack(">ii", data[pos:pos + 8])
        pos += 8
        content = data[pos:pos + content_words * 2]
        pos += content_words * 2
        stype = struct.unpack("<i", content[:4])[0]
        row = attrs[idx] if idx < len(attrs) else {}
        idx += 1
        if stype not in _POLYGON_TYPES:
            continue  # enregistrement nul
        n_parts, n_points = struct.unpack("<ii", content[36:44])
        parts = struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts])
        off = 44 + 4 * n_parts
        pts = np.frombuffer(content, dtype="<f8", count=2 * n_points, offset=off).reshape(n_points, 2)
        bounds = list(parts) + [n_points]
        rings = [pts[bounds[i]:bounds[i + 1]].copy() for i in range(n_parts) if bounds[i + 1] - bounds[i] >= 3]
        shapes.append(Shape(rings=rings, attrs=row))
    return shapes
