import math
import unittest
from unittest import mock

import pandas as pd

from catmap import runoff as rs

INF = float("inf")
T = {"slope": 5.0, "twi": 8.0, "lupa": 0.0, "curv": 0.01, "cn": 80.0}
V_HIGH = {"slope": 15.0, "twi": 10.0, "upa": 9.0, "lupa": math.log10(9.0), "curv": -0.02, "lcfav": 1.0, "cn": 90.0, "hsg": 4, "ksat": 1.0, "geom": 9}
V_LOW = {"slope": 1.0, "twi": 3.0, "upa": 0.5, "lupa": math.log10(0.5), "curv": 0.0, "lcfav": 0.0, "cn": 60.0, "hsg": 1, "ksat": 50.0, "geom": 3}


def hist(values, weights=None):
    w = weights or [1] * len(values)
    return {"bucketMeans": list(values), "histogram": list(w)}


class TestThresholds(unittest.TestCase):
    def test_otsu_separates_two_groups(self):
        h = hist([1, 2, 3, 20, 21, 22], [10, 10, 10, 10, 10, 10])
        t = rs.otsu_threshold(h, min_range=1.0)
        self.assertTrue(3 < t < 20)

    def test_otsu_none_when_uniform_or_small(self):
        self.assertIsNone(rs.otsu_threshold(hist([5, 5.2, 5.4], [100, 100, 100]), min_range=1.0))
        self.assertIsNone(rs.otsu_threshold(hist([1, 50], [2, 3]), min_range=1.0))      # 5 pixels < MIN_PIXELS
        self.assertIsNone(rs.otsu_threshold(None))
        self.assertIsNone(rs.otsu_threshold({"bucketMeans": [], "histogram": []}))

    def test_zone_thresholds_inf_for_uniform_zone(self):
        t = rs.zone_thresholds({"slope": hist([1, 2, 3, 30, 31], [20] * 5), "twi": hist([5, 5.1], [100, 100])})
        self.assertTrue(math.isfinite(t["slope"]))
        self.assertEqual(t["twi"], INF)
        self.assertNotIn("cn", t)


class TestScore(unittest.TestCase):
    def test_all_favourable_classic(self):
        s = rs.score_point(V_HIGH, T)
        self.assertEqual(s["prod"], 1.0)
        self.assertEqual(s["transfert"], 1.0)
        self.assertAlmostEqual(s["accum"], 3 / 5)                  # pente forte et courbure convexe : défavorables à l'accumulation
        self.assertAlmostEqual(s["globale"], (1 + 1 + 0.6) / 3)
        self.assertEqual(s["classe"], 1 + int(s["globale"] * 5))
        self.assertAlmostEqual(s["irip"], round(s["globale"] * 5, 2))
        self.assertEqual(s["irip3"], 1)

    def test_all_unfavourable(self):
        v = dict(V_LOW)
        s = rs.score_point(v, T)
        self.assertEqual(s["prod"], 0.0)
        self.assertEqual(s["transfert"], 0.0)
        self.assertEqual(s["classe"], 1 + min(4, int(s["globale"] * 5)))
        self.assertEqual(s["irip3"], 0)

    def test_no_data_gives_none(self):
        s = rs.score_point({}, {})
        self.assertIsNone(s["globale"])
        self.assertIsNone(s["classe"])
        self.assertIsNone(s["irip3"])

    def test_uniform_zone_has_no_strong_factor(self):
        t = {"slope": INF, "twi": INF, "lupa": INF, "curv": INF, "cn": INF}
        s = rs.score_point(V_HIGH, t)
        self.assertEqual(s["transfert"], 0.0)

    def test_soil_methods(self):
        self.assertEqual(rs.soil_factor({"cn": 90}, {"cn": 80}, "cn"), 1)
        self.assertEqual(rs.soil_factor({"cn": 70}, {"cn": 80}, "cn"), 0)
        self.assertEqual(rs.soil_factor({"hsg": 4}, {}, "hsg"), 1)
        self.assertEqual(rs.soil_factor({"hsg": 24}, {}, "hsg"), 1)
        self.assertEqual(rs.soil_factor({"hsg": 2}, {}, "hsg"), 0)
        self.assertEqual(rs.soil_factor({"ksat": 1.0}, {}, "ksat"), 1)
        self.assertEqual(rs.soil_factor({"ksat": 20.0}, {}, "ksat"), 0)
        self.assertIsNone(rs.soil_factor({}, {}, "hsg"))
        with self.assertRaises(ValueError):
            rs.soil_factor({}, {}, "x")

    def test_geomorphon_scores_and_weight(self):
        scores = {1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 2, 8: 2, 9: 3, 10: 3}
        self.assertEqual(rs.GEOM_SCORE, scores)
        v = dict(V_HIGH, geom=9)
        s = rs.score_point(v, T, accum="geom")
        self.assertAlmostEqual(s["accum"], (1 + 1 + 3) / 5)
        s0 = rs.score_point(dict(V_HIGH, geom=3), T, accum="geom")
        self.assertAlmostEqual(s0["accum"], 2 / 5)

    def test_fixed_area(self):
        v = dict(V_HIGH, upa=0.01, lupa=-2.0)
        self.assertLess(rs.score_point(v, T, area="fixed")["transfert"], 1.0)
        v2 = dict(V_HIGH, upa=0.5, lupa=-0.3)
        self.assertEqual(rs.score_point(v2, dict(T, lupa=INF), area="fixed")["transfert"], 1.0)


class TestEstimate(unittest.TestCase):
    def setUp(self):
        rs._LAYER_OK.clear()
        self.calls = []

    def _run(self, df, zone=None, **kw):
        def fake_batch(ee, layers, basins, z, pts, names, hist_names):
            self.calls.append([k for k, _, _ in pts])
            out = {}
            for k, lo, la in pts:
                out[k] = ({}, {}, -1, 0.0) if la < 0 else (dict(V_HIGH, upa=9.0), dict(T), 123 if la < 44 else 456, 55.5)
            return out
        with mock.patch.object(rs.ch, "init_ee"), mock.patch.object(rs.ch, "_ee", return_value=mock.MagicMock()), \
                mock.patch.object(rs, "layers_available", return_value={"worldcover": True, "gcn": False, "geom": True}), \
                mock.patch.object(rs, "build_layers", return_value=object()), mock.patch.object(rs, "_process_batch", side_effect=fake_batch):
            return rs.estimate(df, "lat", "lon", zone, "proj", {"x": 1}, **kw)

    def test_basin_dedup_chunks_and_status(self):
        df = pd.DataFrame({"lat": [43.0, 43.00001, 44.0, -10.0, "x"], "lon": [7.0, 7.00001, 7.0, 7.0, 1]})
        out, s = self._run(df, {"kind": "basin", "level": 11}, chunk=1, workers=2)
        self.assertEqual(s["pixels_uniques"], 3)
        self.assertEqual(s["lots"], 3)
        self.assertEqual(out[rs.STATUS].iloc[0], "ok")
        self.assertEqual(out[rs.COLUMNS["classe"]].iloc[0], out[rs.COLUMNS["classe"]].iloc[1])
        self.assertEqual(out[rs.ZONE["id"]].iloc[0], 123)
        self.assertEqual(out[rs.ZONE["id"]].iloc[2], 456)
        self.assertEqual(s["zones"], 2)
        self.assertTrue(out[rs.STATUS].iloc[3].startswith("hors terre"))
        self.assertTrue(pd.isna(out[rs.COLUMNS["globale"]].iloc[3]))
        self.assertEqual(out[rs.STATUS].iloc[4], "coordonnées invalides")
        self.assertEqual(s["hors_terre"], 1)
        self.assertEqual(s["jeux_absents"], ["gcn"])
        self.assertIn(rs.COLUMNS["irip"], out.columns)
        self.assertEqual(out[rs.COLUMNS["irip3"]].iloc[0], 1)

    def test_default_zone_is_basin_11(self):
        out, s = self._run(pd.DataFrame({"lat": [43.0], "lon": [7.0]}))
        self.assertEqual(s["zone"], {"kind": "basin", "level": 11})

    def test_window_mode_and_methods(self):
        out, s = self._run(pd.DataFrame({"lat": [43.0], "lon": [7.0]}), {"kind": "window", "radius_km": 5.0}, soil="hsg", accum="geom", area="fixed")
        self.assertEqual(s["zone"]["radius_km"], 5.0)
        self.assertEqual((s["sol"], s["accumulation"], s["aire"]), ("hsg", "geom", "fixed"))
        self.assertEqual(out[rs.ZONE["t_area"]].iloc[0], rs.AREA_FIXED_KM2)

    def test_invalid_arguments(self):
        df = pd.DataFrame({"lat": [1.0], "lon": [1.0]})
        with self.assertRaises(ValueError):
            rs.estimate(df, "lat", "lon", {"kind": "window", "radius_km": 99})
        with self.assertRaises(ValueError):
            rs.estimate(df, "lat", "lon", {"kind": "basin", "level": 13})
        with self.assertRaises(ValueError):
            rs.estimate(df, "lat", "lon", {"kind": "autre"})
        with self.assertRaises(ValueError):
            rs.estimate(df, "lat", "lon", None, soil="inconnu")

    def test_chunk_failure_is_reported_not_fatal(self):
        df = pd.DataFrame({"lat": [43.0, 44.0], "lon": [7.0, 7.0]})

        def boom(ee, layers, basins, z, pts, names, hist_names):
            if pts[0][0] == 0:
                raise RuntimeError("Computation timed out")
            return {k: (dict(V_HIGH), dict(T), 1, 10.0) for k, _, _ in pts}
        with mock.patch.object(rs.ch, "init_ee"), mock.patch.object(rs.ch, "_ee", return_value=mock.MagicMock()), \
                mock.patch.object(rs, "layers_available", return_value={"worldcover": True, "gcn": True}), \
                mock.patch.object(rs, "build_layers", return_value=object()), mock.patch.object(rs, "_process_batch", side_effect=boom):
            out, s = rs.estimate(df, "lat", "lon", None, "p", {"x": 1}, chunk=1, workers=1)
        self.assertEqual(s["echecs"], 1)
        self.assertTrue(out[rs.STATUS].iloc[0].startswith("échec Earth Engine"))
        self.assertEqual(out[rs.STATUS].iloc[1], "ok")

    def test_sea_point_is_expected_in_diagnose(self):
        def fake_est(df, *a, **k):
            out = df.copy()
            out[rs.STATUS] = ["hors terre ou hors couverture"] * len(df)
            return out, {}
        with mock.patch.object(rs.ch, "init_ee"), mock.patch.object(rs.ch, "_ee", return_value=mock.MagicMock()), \
                mock.patch.object(rs, "layers_available", return_value={"worldcover": True, "gcn": True, "hsg": True, "ksat": True, "geom": True}), \
                mock.patch.object(rs, "estimate", side_effect=fake_est):
            rows = rs.diagnose("p", {"x": 1}, {"kind": "window", "radius_km": 2.0})
        sea = [r for r in rows if "mer" in r["étape"]][0]
        self.assertEqual(sea["résultat"], "OK")
        self.assertTrue(sea["détail"].startswith("attendu"))

    def test_diagnose_reports_auth_failure_without_secret(self):
        with mock.patch.object(rs.ch, "init_ee", side_effect=rs.TileError("initialisation Earth Engine impossible")):
            rows = rs.diagnose("p", {"private_key": "-----BEGIN PRIVATE KEY-----"})
        self.assertEqual(rows[0]["résultat"], "ÉCHEC")
        self.assertNotIn("PRIVATE KEY", str(rows))


class TestLayers(unittest.TestCase):
    def test_missing_merit_band_is_an_error(self):
        rs._LAYER_OK.clear()
        rs._ASSET_KIND.clear()
        ee = mock.MagicMock()
        ee.Image.return_value.bandNames.return_value.getInfo.return_value = ["dir"]
        with self.assertRaises(rs.TileError):
            rs.layers_available(ee, {"worldcover"})


if __name__ == "__main__":
    unittest.main()
