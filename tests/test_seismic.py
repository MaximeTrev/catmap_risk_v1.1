import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from catmap import seismic as sm
from tests.test_coastalhazard import make_tif, Resp


def pga_tif(path):
    arr = np.full((100, 100), 0.05, dtype="float32")        # couvre lon 20–30, lat 50–60
    arr[0:10, 0:10] = 0.4                                      # coin nord-ouest : forte PGA
    arr[90:, 90:] = -9999                                      # coin sud-est : sans donnée
    return make_tif(path, arr)


class Sess:
    def __init__(self, body, status=200):
        self.body, self.status, self.urls = body, status, []

    def get(self, url, **kw):
        self.urls.append(url)
        return Resp(self.status, self.body)


class TestSeismic(unittest.TestCase):
    def test_columns_and_classes(self):
        self.assertEqual(sm.column_name(475), "seisme_RP475_pga")
        self.assertEqual([sm.pga_class(v) for v in (0.05, 0.1, 0.2, 0.5)], [0, 1, 2, 3])
        self.assertIsNone(sm.pga_class(float("nan")))

    def test_download_zip_extract_and_sample(self):
        with tempfile.TemporaryDirectory() as t:
            raw = pga_tif(Path(t) / "src.tif")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("v2023_1_pga_475_rock_3min.tif", raw)
            sess = Sess(buf.getvalue())
            cache = Path(t) / "cache"
            df = pd.DataFrame({"name": list("ABCDE"), "lat": ["59,5", 55.0, 50.2, 40.0, 59.5], "long": ["20,5", 25.0, 29.8, 25.0, 60.0]})
            out, summ = sm.estimate_pga(df, "lat", "long", "gem", [475], session=sess, cache_dir=cache)
            self.assertEqual(len(sess.urls), 1)
            self.assertAlmostEqual(out.loc[0, "seisme_RP475_pga"], 0.4, places=3)        # virgule décimale acceptée
            self.assertAlmostEqual(out.loc[1, "seisme_RP475_pga"], 0.05, places=3)
            self.assertIn("sans donnée", out.loc[2, "seisme_statut"])
            self.assertIn("hors du raster", out.loc[3, "seisme_statut"])
            self.assertEqual(summ["ok"], 2)
            sm.estimate_pga(df, "lat", "long", "gem", [475], session=sess, cache_dir=cache)
            self.assertEqual(len(sess.urls), 1)                                          # cache réutilisé
            rows = sm.diagnose(sess, cache)
            self.assertEqual(rows[0]["résultat"], "OK")

    def test_errors(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(sm.SeismicError):
                sm.gem_path(Path(t), Sess(b"", 404), data_dir=Path(t) / "none")
            with self.assertRaises(sm.SeismicError):
                sm.gem_path(Path(t) / "c2", Sess(b"pas un zip"), data_dir=Path(t) / "none")
            with self.assertRaises(sm.SeismicError):
                sm.estimate_pga(pd.DataFrame({"lat": [1], "long": [1]}), "lat", "long", "eshm20", [475])
            with self.assertRaises(sm.SeismicError):
                sm.estimate_pga(pd.DataFrame({"lat": [1], "long": [1]}), "lat", "long", "gem", [2475])

    def test_local_file_has_priority(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "data"
            d.mkdir()
            pga_tif(d / "v2023_1_pga_475_rock_3min.tif")
            self.assertEqual(sm.gem_path(Path(t) / "c", Sess(b"", 500), data_dir=d).name, "v2023_1_pga_475_rock_3min.tif")

    def test_vulnerability_curve(self):
        c = sm.vulnerability_curve(0.6, 0.6)
        self.assertEqual(float(c.at(0.0)), 0.0)
        self.assertAlmostEqual(float(c.at(0.6)), 0.5, delta=0.01)
        self.assertLess(float(c.at(0.2)), float(c.at(0.8)))
        with self.assertRaises(ValueError):
            sm.vulnerability_curve(0, 0.6)

    def test_openapi_summary(self):
        spec = {"paths": {"/v1/curves": {"get": {"parameters": [{"name": "imt", "required": True, "schema": {"type": "string", "enum": ["PGA", "SA(0.2)"]}}]}},
                          "/v1/models": {"get": {"summary": "liste"}}}}
        rows = sm.summarize_openapi(spec)
        self.assertEqual(rows[0]["paramètre"], "imt")
        self.assertIn("PGA", rows[0]["valeurs"])
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
