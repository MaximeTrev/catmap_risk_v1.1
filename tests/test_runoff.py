import unittest
from unittest import mock

import pandas as pd

from catmap import runoff as rs

P = {"slope_p75": 10.0, "slope_p25": 2.0, "upa_p75": 5.0, "conc_p75": 3.0, "twi_p75": 8.0, "cn_p75": 80.0}
V_HIGH = {"slope": 15.0, "conc": 5.0, "twi": 10.0, "upa": 9.0, "lcfav": 1.0, "cn": 90.0}
V_LOW = {"slope": 1.0, "conc": 0.0, "twi": 3.0, "upa": 0.5, "lcfav": 0.0, "cn": 60.0}


class TestScore(unittest.TestCase):
    def test_all_favourable(self):
        s = rs.score_point(V_HIGH, P)
        self.assertEqual(s["prod"], 1.0)
        self.assertEqual(s["transfert"], 1.0)
        self.assertAlmostEqual(s["accum"], 4 / 5)                  # pente forte : défavorable à l'accumulation
        self.assertAlmostEqual(s["globale"], (1 + 1 + 0.8) / 3)
        self.assertEqual(s["classe"], 1 + int(s["globale"] * 5))

    def test_all_unfavourable(self):
        s = rs.score_point(V_LOW, P)
        self.assertEqual(s["prod"], 0.0)
        self.assertEqual(s["transfert"], 0.0)
        self.assertAlmostEqual(s["accum"], 1 / 5)                  # seule la pente faible est favorable
        self.assertEqual(s["classe"], 1)

    def test_equal_weights_and_missing_factor(self):
        s = rs.score_point({"slope": 15.0, "conc": 5.0, "twi": 10.0, "upa": 9.0, "lcfav": None, "cn": None}, P)   # un seul facteur de production
        self.assertEqual(s["prod"], 1.0)                           # facteurs absents écartés, pas comptés défavorables
        self.assertAlmostEqual(s["globale"], (s["prod"] + s["transfert"] + s["accum"]) / 3)

    def test_no_data_gives_none(self):
        s = rs.score_point({}, {})
        self.assertIsNone(s["globale"])
        self.assertIsNone(s["classe"])

    def test_ties_are_not_favourable(self):                         # terrain uniforme : valeur = quartile -> pas favorable
        v = {"slope": 5.0, "conc": 1.0, "twi": 5.0, "upa": 2.0, "lcfav": 0.0, "cn": 70.0}
        p = {"slope_p75": 5.0, "slope_p25": 5.0, "upa_p75": 2.0, "conc_p75": 1.0, "twi_p75": 5.0, "cn_p75": 70.0}
        s = rs.score_point(v, p)
        self.assertEqual((s["prod"], s["transfert"], s["accum"]), (0.0, 0.0, 0.0))

    def test_classes_bounds(self):
        for g, c in ((0.0, 1), (0.19, 1), (0.2, 2), (0.99, 5), (1.0, 5)):
            self.assertEqual(1 + min(4, int(g * 5)), c)


class TestEstimate(unittest.TestCase):
    def setUp(self):
        rs._LAYER_OK.clear()
        self.calls = []

    def _run(self, df, **kw):
        def fake_reduce(ee, layers, pts, radius, names):
            self.calls.append([k for k, _, _ in pts])
            out = {}
            for k, lo, la in pts:
                out[k] = ({}, {}) if la < 0 else (dict(V_HIGH), dict(P))        # latitude négative = « mer » dans ce test
            return out
        with mock.patch.object(rs.ch, "init_ee"), mock.patch.object(rs.ch, "_ee", return_value=mock.MagicMock()), \
                mock.patch.object(rs, "layers_available", return_value={"worldcover": True, "gcn": False}), \
                mock.patch.object(rs, "build_layers", return_value=object()), mock.patch.object(rs, "_reduce_chunk", side_effect=fake_reduce):
            return rs.estimate(df, "lat", "lon", 2.0, "proj", {"x": 1}, **kw)

    def test_dedup_chunks_and_status(self):
        df = pd.DataFrame({"lat": [43.0, 43.00001, 44.0, -10.0, "x"], "lon": [7.0, 7.00001, 7.0, 7.0, 1]})
        out, s = self._run(df, chunk=1, workers=2)
        self.assertEqual(s["pixels_uniques"], 3)                    # les deux premiers points tombent sur le même pixel
        self.assertEqual(s["lots"], 3)
        self.assertEqual(out[rs.STATUS].iloc[0], "ok")
        self.assertEqual(out[rs.COLUMNS["classe"]].iloc[0], out[rs.COLUMNS["classe"]].iloc[1])
        self.assertTrue(out[rs.STATUS].iloc[3].startswith("hors terre"))
        self.assertTrue(pd.isna(out[rs.COLUMNS["globale"]].iloc[3]))
        self.assertEqual(out[rs.STATUS].iloc[4], "coordonnées invalides")
        self.assertEqual(s["hors_terre"], 1)
        self.assertEqual(s["jeux_absents"], ["gcn"])
        self.assertEqual(s["facteurs"], ["slope", "conc", "twi", "upa", "lcfav"])

    def test_chunk_failure_is_reported_not_fatal(self):
        df = pd.DataFrame({"lat": [43.0, 44.0], "lon": [7.0, 7.0]})

        def boom(ee, layers, pts, radius, names):
            if pts[0][0] == 0:
                raise RuntimeError("Computation timed out")
            return {k: (dict(V_HIGH), dict(P)) for k, _, _ in pts}
        with mock.patch.object(rs.ch, "init_ee"), mock.patch.object(rs.ch, "_ee", return_value=mock.MagicMock()), \
                mock.patch.object(rs, "layers_available", return_value={"worldcover": True, "gcn": True}), \
                mock.patch.object(rs, "build_layers", return_value=object()), mock.patch.object(rs, "_reduce_chunk", side_effect=boom):
            out, s = rs.estimate(df, "lat", "lon", 2.0, "p", {"x": 1}, chunk=1, workers=1)
        self.assertEqual(s["echecs"], 1)
        self.assertTrue(out[rs.STATUS].iloc[0].startswith("échec Earth Engine"))
        self.assertEqual(out[rs.STATUS].iloc[1], "ok")

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            rs.estimate(pd.DataFrame({"lat": [1.0], "lon": [1.0]}), "lat", "lon", 99.0)

    def test_diagnose_reports_auth_failure_without_secret(self):
        with mock.patch.object(rs.ch, "init_ee", side_effect=rs.TileError("initialisation Earth Engine impossible")):
            rows = rs.diagnose("p", {"private_key": "-----BEGIN PRIVATE KEY-----"})
        self.assertEqual(rows[0]["résultat"], "ÉCHEC")
        self.assertNotIn("PRIVATE KEY", str(rows))


class TestLayers(unittest.TestCase):
    def test_missing_merit_band_is_an_error(self):
        rs._LAYER_OK.clear()
        ee = mock.MagicMock()
        ee.Image.return_value.bandNames.return_value.getInfo.return_value = ["dir"]
        with self.assertRaises(rs.TileError):
            rs.layers_available(ee)

    def test_optional_layer_failure_is_tolerated(self):
        rs._LAYER_OK.clear()
        ee = mock.MagicMock()
        ee.Image.return_value.bandNames.return_value.getInfo.side_effect = [["elv", "upa"], RuntimeError("not found")]
        ee.ImageCollection.return_value.first.return_value.bandNames.return_value.getInfo.return_value = ["Map"]
        use = rs.layers_available(ee)
        self.assertEqual(use, {"worldcover": True, "gcn": False})


if __name__ == "__main__":
    unittest.main()
