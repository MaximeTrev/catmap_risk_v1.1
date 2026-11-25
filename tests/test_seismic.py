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


class TestCompactGem(unittest.TestCase):
    def build(self, t, strips=False):
        p = Path(t) / "v2023_1_pga_475_rock_3min.tif"
        if strips:
            arr = np.full((60, 100), 0.07, dtype="float32")
            extra = [(33550, "d", 3, (0.1, 0.1, 0.0), True), (33922, "d", 6, (0, 0, 0, 20.0, 60.0, 0), True)]
            tifffile.imwrite(str(p), arr, compression="deflate", photometric="minisblack", rowsperstrip=1, extratags=extra)
        else:
            pga_tif(p)
        out = Path(t) / "data" / sm.COMPACT_NAME
        out.parent.mkdir(exist_ok=True)
        info = sm.build_compact(p, out, progress=lambda *_: None)
        return p, out, info

    def test_build_and_sample_match_raster(self):
        with tempfile.TemporaryDirectory() as t:
            p, out, info = self.build(t)
            self.assertLess(info["taille_mo"], 1)
            pts = [(59.5, 20.5), (55.0, 25.0), (50.2, 29.8), (40.0, 25.0)]       # forte PGA ; courante ; sans donnée ; hors image
            raw = sm.fhz.sample_geotiff(p, [x[1] for x in pts], [x[0] for x in pts])
            cmp_ = sm.sample_compact(out, [x[1] for x in pts], [x[0] for x in pts])
            for a, b in zip(raw, cmp_):
                if a is None:
                    self.assertIsNone(b)
                else:
                    self.assertAlmostEqual(a, b, delta=1.1e-4)               # précision du stockage : 1e-4 g
            self.assertEqual(sm.compact_extent(out), (20.0, 50.0, 30.0, 60.0))

    def test_strips_of_one_row(self):
        with tempfile.TemporaryDirectory() as t:
            _p, out, _ = self.build(t, strips=True)
            self.assertAlmostEqual(sm.sample_compact(out, [25.0], [55.0])[0], 0.07, places=4)

    def test_estimate_uses_compact_without_network_and_raw_option(self):
        with tempfile.TemporaryDirectory() as t:
            p, out, _ = self.build(t)

            class NoNet:
                def get(self, *a, **k):
                    raise AssertionError("aucun accès réseau attendu")
            df = pd.DataFrame({"name": list("ABCD"), "lat": ["59,5", 55.0, 50.2, 40.0], "long": ["20,5", 25.0, 29.8, 25.0]})
            o, s = sm.estimate_pga(df, "lat", "long", "gem", [475], session=NoNet(), cache_dir=Path(t) / "c", data_dir=out.parent)
            self.assertEqual(s["source"], "fichier précalculé")
            self.assertAlmostEqual(o.loc[0, "seisme_RP475_pga"], 0.4, places=3)
            self.assertIn("sans donnée", o.loc[2, "seisme_statut"])
            self.assertIn("hors du raster", o.loc[3, "seisme_statut"])
            self.assertEqual(s["ok"], 2)
            # données brutes : le fichier précalculé est ignoré, le raster d'origine est lu
            o2, s2 = sm.estimate_pga(df, "lat", "long", "gem", [475], cache_dir=Path(t) / "c", data_dir=Path(t), use_compact=False, tif=p)
            self.assertEqual(s2["source"], "raster GEM brut")
            self.assertTrue(np.allclose(o["seisme_RP475_pga"].dropna(), o2["seisme_RP475_pga"].dropna(), atol=2e-4))
            rows = sm.diagnose(NoNet(), Path(t) / "c", out.parent)            # sans « brut » : aucun téléchargement
            self.assertEqual(rows[0]["étape"], "fichier précalculé")
            self.assertEqual(rows[0]["résultat"], "OK")
            self.assertEqual(rows[-1]["étape"], "raster brut")

    def test_compact_in_parent_data_folder_and_incomplete(self):
        with tempfile.TemporaryDirectory() as t:
            _p, out, _ = self.build(t)
            (out.parent / "seisme").mkdir()
            self.assertEqual(sm.compact_path(out.parent / "seisme"), out)
            np.savez(Path(t) / "bad" / "x.npz" if False else Path(t) / sm.COMPACT_NAME, pga=np.zeros((2, 2), np.uint16))
            with self.assertRaises(sm.SeismicError):
                sm.sample_compact(Path(t) / sm.COMPACT_NAME, [0], [0])

    def test_standalone_script_gives_same_file(self):
        import importlib.util
        with tempfile.TemporaryDirectory() as t:
            p, out, _ = self.build(t)
            folder = Path(t) / "gem"
            folder.mkdir()
            with zipfile.ZipFile(folder / "GEM-GSHM_PGA-475y-rock_v2023.zip", "w") as z:     # le script sait extraire d'un zip
                z.write(p, p.name)
            spec = importlib.util.spec_from_file_location("bgc", Path(__file__).resolve().parent.parent / "tools" / "build_gem_compact.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            res = m.main([str(folder)])
            self.assertIsNotNone(res)
            a, b = np.load(folder / "gem_pga475_compact.npz"), np.load(out)
            self.assertTrue(all(np.array_equal(a[k], b[k]) for k in b.files))
            self.assertIsNone(m.main([str(Path(t) / "inexistant")]))                                # message clair, sans exception
            empty = Path(t) / "vide"
            empty.mkdir()
            self.assertIsNone(m.main([str(empty)]))


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
            n_urls = len(sess.urls)                                                      # API Zenodo (illisible ici) + adresse devinée
            self.assertGreaterEqual(n_urls, 1)
            self.assertAlmostEqual(out.loc[0, "seisme_RP475_pga"], 0.4, places=3)        # virgule décimale acceptée
            self.assertAlmostEqual(out.loc[1, "seisme_RP475_pga"], 0.05, places=3)
            self.assertIn("sans donnée", out.loc[2, "seisme_statut"])
            self.assertIn("hors du raster", out.loc[3, "seisme_statut"])
            self.assertEqual(summ["ok"], 2)
            sm.estimate_pga(df, "lat", "long", "gem", [475], session=sess, cache_dir=cache)
            self.assertEqual(len(sess.urls), n_urls)                                     # cache réutilisé
            rows = sm.diagnose(sess, cache)
            self.assertEqual(rows[0]["résultat"], "OK")

    def test_errors(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(sm.SeismicError):
                sm.gem_path(Path(t), Sess(b"", 404), data_dir=Path(t) / "none")
            with self.assertRaises(sm.SeismicError):
                sm.gem_path(Path(t) / "c2", Sess(b"pas un zip"), data_dir=Path(t) / "none")
            with self.assertRaises(sm.SeismicError):
                sm.estimate_pga(pd.DataFrame({"lat": [1], "long": [1]}), "lat", "long", "gem", [2475])

    def test_html_instead_of_zip_gives_readable_error(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(sm.SeismicError) as cm:
                sm.gem_path(Path(t) / "c", Sess(b"<html>blocked</html>"), data_dir=Path(t) / "none")
            self.assertIn("ni zip ni TIFF", str(cm.exception))

    def test_direct_tif_download(self):
        with tempfile.TemporaryDirectory() as t:
            raw = pga_tif(Path(t) / "src.tif")
            p = sm.gem_path(Path(t) / "c", Sess(raw), data_dir=Path(t) / "none")
            self.assertTrue(p.name.endswith(".tif") or p.suffix == ".tif")

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

    def test_strip_of_one_row_is_supported(self):
        import tifffile
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "v2023_1_pga_475_rock_3min.tif"
            arr = np.full((60, 100), 0.07, dtype="float32")
            extra = [(33550, "d", 3, (0.1, 0.1, 0.0), True), (33922, "d", 6, (0, 0, 0, 20.0, 60.0, 0), True)]
            tifffile.imwrite(str(p), arr, compression="deflate", photometric="minisblack", rowsperstrip=1, extratags=extra)   # bandes d'1 ligne
            out, summ = sm.estimate_pga(pd.DataFrame({"lat": [55.0], "long": [25.0]}), "lat", "long", "gem", [475], tif=p)
            self.assertAlmostEqual(out.loc[0, "seisme_RP475_pga"], 0.07, places=3)

    def test_zip_in_data_dir(self):
        with tempfile.TemporaryDirectory() as t:
            raw = pga_tif(Path(t) / "src.tif")
            d = Path(t) / "data"
            d.mkdir()
            with zipfile.ZipFile(d / "GEM-GSHM_PGA-475y-rock_v2023.zip", "w") as z:
                z.writestr("v2023_1_pga_475_rock_3min.tif", raw)
            p = sm.gem_path(Path(t) / "cache", Sess(b"", 500), data_dir=d)
            self.assertTrue(p.exists())

    def test_eshm_parsing_and_interpolation(self):
        iml = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
        poe = [0.2, 0.03, 0.008, 0.0021, 0.0004, 0.00005]
        for js in ({"imls": iml, "poes": poe}, {"curves": [{"x": iml, "y": poe}]}, [{"iml": a, "poe": b} for a, b in zip(iml, poe)],
                   {"data": [[a, b] for a, b in zip(iml, poe)]}):
            x, y = sm.parse_curve(js)
            self.assertAlmostEqual(sm.pga_at_rp(x, y, 475), 0.2, delta=0.02)
        self.assertIsNone(sm.pga_at_rp(*sm.parse_curve({"imls": iml, "poes": poe}), 1_000_000))   # hors courbe : pas d'extrapolation
        with self.assertRaises(sm.SeismicError):
            sm.parse_curve({"foo": "bar"})
        self.assertEqual(sm.pick_eshm_model([{"id": 3, "name": "SHARE"}, {"id": 12, "title": "ESHM20 Europe"}])[0], 12)
        self.assertIsNone(sm.pick_eshm_model([{"id": 3, "name": "SHARE"}]))

    def test_eshm_real_response_format(self):
        pts = [{"iml": 0.0005, "poe": 0.9998}, {"iml": 0.05, "poe": 0.3}, {"iml": 0.1, "poe": 0.1}, {"iml": 0.2, "poe": 0.0235},
               {"iml": 0.4, "poe": 0.004}, {"iml": 0.8, "poe": 0.0004}]
        js = {"curves": [{"lon": 6.18, "lat": 46.2, "model_id": 81, "imt": "PGA", "aggregation_type": "quantile", "aggregation_level": 0.16,
                          "investigation_time": 50, "points": [dict(p, poe=p["poe"] / 2) for p in pts]},
                         {"lon": 6.18, "lat": 46.2, "model_id": 81, "imt": "PGA", "aggregation_type": "arithmetic", "aggregation_level": 0.5,
                          "investigation_time": 50, "points": pts}]}
        iml, poe = sm.parse_curve(js)
        self.assertAlmostEqual(poe[2], 1 - 0.9 ** (1 / 50), places=6)                  # poe sur 50 ans -> probabilité annuelle
        v = sm.pga_at_rp(iml, poe, 475)                                                  # 10 % en 50 ans = 475 ans -> 0,1 g
        self.assertAlmostEqual(v, 0.1, delta=0.005)
        self.assertEqual(sm.pick_curve(js["curves"])["aggregation_type"], "arithmetic")  # moyenne retenue, pas le quantile
        self.assertAlmostEqual(sm.pga_at_rp(iml, poe, 2475), 0.2, delta=0.02)            # 2 % en 50 ans -> ~0,2 g

    def test_eshm_estimate_cache_and_domain(self):
        calls = []

        def fake_get(path, params, session=None, timeout=40):
            calls.append(path)
            if path == "/v1/models":
                return [{"id": 7, "name": "ESHM20"}] if params["lat"] > 30 else []
            if params["lat"] < 30:
                raise sm.SeismicError("EFEHR /v1/curves : HTTP 404 hors du domaine du modèle")
            return {"imls": [0.01, 0.05, 0.1, 0.2, 0.4, 0.8], "poes": [0.2, 0.03, 0.008, 0.0021, 0.0004, 0.00005]}
        from unittest import mock
        df = pd.DataFrame({"lat": ["46,2", 46.21, -33.0], "long": [6.15, 6.16, 151.0]})
        with tempfile.TemporaryDirectory() as t, mock.patch.object(sm, "_get_json", side_effect=fake_get):
            out, summ = sm.estimate_pga(df, "lat", "long", "eshm20", [100, 475, 2475], cache_dir=Path(t))
            self.assertEqual(summ["ok"], 2)
            self.assertIn("domaine", out.loc[2, "seisme_statut"])
            n = len(calls)
            sm.estimate_pga(df, "lat", "long", "eshm20", [475], cache_dir=Path(t))
            self.assertEqual(len(calls), n)                                           # mailles en cache : aucune nouvelle requête
            self.assertLess(out.loc[0, "seisme_RP100_pga"], out.loc[0, "seisme_RP475_pga"])

    def test_openapi_summary(self):
        spec = {"paths": {"/v1/curves": {"get": {"parameters": [{"name": "imt", "required": True, "schema": {"type": "string", "enum": ["PGA", "SA(0.2)"]}}]}},
                          "/v1/models": {"get": {"summary": "liste"}}}}
        rows = sm.summarize_openapi(spec)
        self.assertEqual(rows[0]["paramètre"], "imt")
        self.assertIn("PGA", rows[0]["valeurs"])
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
