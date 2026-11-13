import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from catmap import floodhazard as fh

BASE = fh.BASE_URL


def make_tif(path, arr, tiled=True, nodata="-9999"):
    extra = [(33550, "d", 3, (0.1, 0.1, 0.0), True), (33922, "d", 6, (0, 0, 0, 20.0, 60.0, 0), True)]
    if nodata is not None:
        extra.append((42113, "s", 0, nodata, True))
    kw = dict(tile=(32, 32)) if tiled else dict(rowsperstrip=7)
    tifffile.imwrite(str(path), arr, compression="deflate", photometric="minisblack", extratags=extra, **kw)
    return Path(path).read_bytes()


class Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code, self.content, self.headers = status, body, headers or {}
        self.text = body.decode("utf-8", "ignore") if isinstance(body, bytes) else str(body)

    def json(self):
        return json.loads(self.text)

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def close(self):
        pass


class Sess:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, timeout=None, headers=None, stream=False, **kw):
        self.calls.append((url, dict(headers or {})))
        r = self.routes.get(url)
        if r is None:
            return Resp(404, b"nf")
        if headers and "Range" in headers and r.status_code == 200:
            return Resp(206, r.content[:100], {"Content-Range": f"bytes 0-99/{len(r.content)}"})
        return r


def tile_arrays():
    raw = np.full((100, 100), -9999, "float32")
    raw[10, 20] = 2.5
    raw[10, 21] = 0.4
    rec = np.zeros((100, 100), "uint8")
    rec[10, 20] = 3
    rec[10, 21] = 1
    return raw, rec


GJ = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"ID": 1}, "geometry": {"type": "Polygon", "coordinates": [[[20, 50], [30, 50], [30, 60], [20, 60], [20, 50]]]}},
    {"type": "Feature", "properties": {"ID": 2}, "geometry": {"type": "Polygon", "coordinates": [[[30, 50], [40, 50], [40, 60], [30, 60], [30, 50]]]}},
]}


def routes(folder="RP100", listing=True, missing_reclass=False):
    d = Path(tempfile.mkdtemp())
    raw, rec = tile_arrays()
    r = {BASE + "tile_extents.geojson": Resp(200, json.dumps(GJ).encode())}
    raw_b = make_tif(d / "a.tif", raw, tiled=True)
    rec_b = make_tif(d / "b.tif", rec, tiled=False, nodata=None)
    f = f"{BASE}{folder}/"
    if listing:
        r[f] = Resp(200, ('<a href="ID1_N60_E20_RP100_depth.tif">x</a><a href="ID1_N60_E20_RP100_depth_reclass.tif">y</a>'
                          '<a href="ID2_N60_E30_RP100_depth.tif">z</a>').encode())
    else:
        r[f] = Resp(200, b"<html>no index</html>")
    r[f + "ID1_N60_E20_RP100_depth.tif"] = Resp(200, raw_b)
    if not missing_reclass:
        r[f + "ID1_N60_E20_RP100_depth_reclass.tif"] = Resp(200, rec_b)
    r[f + "ID2_N60_E30_RP100_depth.tif"] = Resp(200, raw_b)
    return r


class T(unittest.TestCase):
    def setUp(self):
        self.cache = Path(tempfile.mkdtemp())

    def test_names_and_classes(self):
        self.assertEqual(fh.column_name(100, "raw"), "debordement_RP100_raw")
        c = lambda v, k="raw": fh.flood_class(v, k)
        self.assertEqual([c(0), c(0.5), c(1), c(2.9), c(3), c(9.9), c(10), c(None)], [0, 1, 2, 2, 3, 3, 4, None])
        self.assertEqual([c(1, "reclass"), c(4, "reclass"), c(0, "reclass")], [1, 4, 0])

    def test_catalog_and_locate(self):
        cat = fh.Catalog(fh.parse_tile_extents(GJ))
        t = cat.locate_many(np.array([55.0, 55.0, 10.0]), np.array([25.0, 35.0, 25.0]))
        self.assertEqual([x.id if x else None for x in t], [1, 2, None])
        gj2 = {"features": [{"properties": {"name": "ID007_N10_E20"}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}]}
        self.assertEqual(fh.parse_tile_extents(gj2)[0].id, 7)

    def test_listing(self):
        lst = fh.parse_listing('<a href="ID12_N60_E20_RP50_depth.tiff">a</a> <a href="ID12_N60_E20_RP50_depth_reclass.tiff">b</a>')
        self.assertEqual(lst, {12: {"raw": "ID12_N60_E20_RP50_depth.tiff", "reclass": "ID12_N60_E20_RP50_depth_reclass.tiff"}})

    def test_sampling_tiled_and_strips(self):
        d = Path(tempfile.mkdtemp())
        raw, rec = tile_arrays()
        make_tif(d / "a.tif", raw, tiled=True)
        make_tif(d / "b.tif", rec, tiled=False, nodata=None)
        lon, lat = [22.05, 22.15, 25.0, 19.0, 22.05], [58.95, 58.95, 55.0, 55.0, 61.0]
        self.assertEqual(fh.sample_geotiff(d / "a.tif", lon, lat), [2.5, float(np.float32(0.4)), None, None, None])
        self.assertEqual(fh.sample_geotiff(d / "b.tif", lon, lat), [3.0, 1.0, 0.0, None, None])   # 0 = valeur lue (sans nodata déclaré)

    def test_estimate_end_to_end(self):
        s = Sess(routes())
        df = pd.DataFrame({"name": ["a", "b", "c", "d"], "lat": [58.95, 55.0, 10.0, "x"], "long": [22.05, 35.0, 25.0, 1]})
        out, summ = fh.estimate_depths(df, "lat", "long", [100], ["raw", "reclass"], session=s, cache_dir=self.cache)
        self.assertEqual(list(out["debordement_RP100_raw"])[:2], [2.5, 0.0])        # point b : tuile 2, sans-donnée -> 0
        self.assertEqual(out.loc[0, "debordement_RP100_reclass"], 3)
        self.assertTrue(pd.isna(out.loc[2, "debordement_RP100_raw"]))
        self.assertEqual(list(out["debordement_tuile"])[:2], ["ID1", "ID2"])
        self.assertTrue(pd.isna(out.loc[2, "debordement_tuile"]))
        self.assertEqual(out.loc[0, "debordement_statut"], "ok")
        self.assertTrue(out.loc[2, "debordement_statut"].startswith("hors tuile"))
        self.assertTrue(out.loc[3, "debordement_statut"].startswith("hors tuile"))
        self.assertEqual((summ["tuiles"], summ["fichiers"], summ["hors_tuile"]), (2, 4, 2))
        # tuile 2 sans fichier reclass listé : échec propre, pas d'exception, statut partiel
        self.assertTrue(out.loc[1, "debordement_statut"].startswith("partiel"))
        # un second passage réutilise le cache disque
        n = len(s.calls)
        fh.estimate_depths(df, "lat", "long", [100], ["raw"], session=s, cache_dir=self.cache)
        self.assertEqual(len([c for c in s.calls[n:] if c[0].endswith(".tif") and "Range" not in c[1]]), 0)

    def test_folder_variants_and_candidate_names(self):
        s = Sess(routes(folder="RP100", listing=False))        # dossier sans index exploitable : noms déduits des coordonnées
        df = pd.DataFrame({"lat": [58.95], "long": [22.05]})
        out, _ = fh.estimate_depths(df, "lat", "long", [100], ["raw"], session=s, cache_dir=self.cache)
        self.assertEqual(out.loc[0, "debordement_RP100_raw"], 2.5)
        r = fh.Resolver(Sess(routes(folder="RP050")))     # variante de nom de dossier à 3 chiffres (RP50 en 404, puis RP050)
        self.assertTrue(r.folder(50).endswith("RP050/"))
        with self.assertRaises(fh.TileError):
            fh.Resolver(Sess(routes(folder="RP100"))).folder(500)     # dossier absent : erreur claire

    def test_failures_are_statuses(self):
        r = routes()
        r[BASE + "RP100/ID1_N60_E20_RP100_depth.tif"] = Resp(500, b"x")
        df = pd.DataFrame({"lat": [58.95], "long": [22.05]})
        out, summ = fh.estimate_depths(df, "lat", "long", [100], ["raw"], session=Sess(r), cache_dir=self.cache)
        self.assertTrue(out.loc[0, "debordement_statut"].startswith("partiel"))
        self.assertTrue(pd.isna(out.loc[0, "debordement_RP100_raw"]))
        self.assertTrue(summ["problemes"])
        bad = Path(self.cache) / "bad.tif"
        bad.write_bytes(b"not a tiff")
        with self.assertRaises(fh.TileError):
            fh.sample_geotiff(bad, [1.0], [1.0])

    def test_catalog_errors(self):
        with self.assertRaises(fh.TileError):
            fh.load_catalog(Sess({}))

    def test_diagnose(self):
        rows = fh.diagnose(100, Sess(routes()))
        self.assertTrue(all(x["résultat"] == "OK" for x in rows), rows)
        self.assertTrue(any("Range" in x["étape"] for x in rows))
        rows = fh.diagnose(100, Sess({}))
        self.assertEqual(rows[0]["résultat"], "ÉCHEC")


if __name__ == "__main__":
    unittest.main()
