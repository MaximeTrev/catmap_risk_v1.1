import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from catmap import coastalhazard as ch
from tests.test_floodhazard import Resp, Sess, make_tif


def arr():
    a = np.full((100, 100), -9999, "float32")
    a[10, 20] = 2.5
    return a


class TestCoastal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_names_and_columns(self):
        self.assertEqual(ch.candidate_names("historical", "nosub", "hist", 10)[0], "inuncoast_historical_nosub_hist_rp0010_0.tif")
        self.assertIn("inuncoast_historical_nosub_hist_rp0010_5.tif", ch.candidate_names("historical", "nosub", "hist", 10))
        self.assertEqual(ch.candidate_names("rcp8p5", "wtsub", "2050", 100)[0], "inuncoast_rcp8p5_wtsub_2050_rp0100_0_perc_50.tif")
        self.assertEqual(ch.column_name(100), "submersion_RP100_raw")

    def test_listing_is_used_first(self):
        xml = ("<ListBucketResult><Key>AqueductFloodTool/download/v2/inuncoast_historical_nosub_hist_rp0100_5.tif</Key>"
               "<Key>AqueductFloodTool/download/v2/inuncoast_historical_nosub_hist_rp0100_0_perc_50.tif</Key></ListBucketResult>")
        sess = Sess({ch.S3_HOST: Resp(200, xml.encode(), {})})
        self.assertEqual(ch.list_remote_names("historical", "nosub", "hist", 100, sess)[0], "inuncoast_historical_nosub_hist_rp0100_0_perc_50.tif")
        self.assertEqual(ch.list_remote_names("historical", "nosub", "hist", 100, Sess({})), [])

    def test_estimate_values_and_partial_failure(self):
        body = make_tif(Path(self.tmp.name) / "src.tif", arr())
        sess = Sess({ch.BASE_URL + "inuncoast_historical_nosub_hist_rp0100_0.tif": Resp(200, body, {"Content-Length": str(len(body))})})
        df = pd.DataFrame({"lat": [58.95, 58.0, "x", 10.0], "lon": [22.05, 22.05, 1, 22.05]})
        out, s = ch.estimate_depths(df, "lat", "lon", [100, 500], "historical", "nosub", "hist", session=sess,
                                    cache_dir=Path(self.tmp.name) / "c")
        self.assertEqual(out["submersion_RP100_raw"].iloc[0], 2.5)
        self.assertEqual(out["submersion_RP100_raw"].iloc[1], 0.0)             # sans-donnée dans l'image : aucune submersion modélisée
        self.assertTrue(pd.isna(out["submersion_RP100_raw"].iloc[2]))                 # coordonnées invalides : rien d'inventé
        self.assertIn("RP500", s["problemes"])                                  # fichier introuvable : signalé, sans arrêter RP100
        self.assertEqual(out["submersion_statut"].iloc[2], "coordonnées invalides")
        self.assertTrue(out["submersion_statut"].iloc[0].startswith("partiel"))
        self.assertEqual(s["invalides"], 1)
        self.assertTrue(pd.isna(out["submersion_RP100_raw"].iloc[3]))          # hors de l'emprise du raster : vide, pas 0
        self.assertTrue(out["submersion_statut"].iloc[3].startswith("hors du raster"))
        self.assertEqual(s["hors_raster"], 1)
        self.assertEqual(s["points_positifs"]["RP100"], 1)

    def test_future_scenario_tries_alternative_names(self):
        body = make_tif(Path(self.tmp.name) / "src.tif", arr())
        sess = Sess({ch.BASE_URL + "inuncoast_rcp4p5_wtsub_2050_rp0100_0.tif": Resp(200, body, {"Content-Length": str(len(body))})})
        url = ch.resolve_url("rcp4p5", "wtsub", "2050", 100, sess)
        self.assertTrue(url.endswith("_0.tif"))
        with self.assertRaises(ch.TileError):
            ch.resolve_url("rcp4p5", "wtsub", "2030", 100, sess)


if __name__ == "__main__":
    unittest.main()
