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


class FakeEE:
    """Simulation minimale de ``ee`` : aucune connexion réseau."""
    IMAGES = [
        {"floodtype": "inuncoast", "climatescenario": "historical", "subsidence": "nosub", "year": 1980, "returnperiod": 100, "projection": "NA", "img_id": "H100"},
        {"floodtype": "inuncoast", "climatescenario": "historical", "subsidence": "nosub", "year": 1980, "returnperiod": "10", "projection": "NA", "img_id": "H10"},
        {"floodtype": "inuncoast", "climatescenario": "rcp8p5", "subsidence": "wtsub", "year": 2050, "returnperiod": 100, "projection": 95, "img_id": "F100_95"},
        {"floodtype": "inunriver", "climatescenario": "historical", "subsidence": "nosub", "year": 1980, "returnperiod": 100, "projection": "NA", "img_id": "RIV"},
    ]
    VALUES = {"H100": {0: 2.5, 1: None}, "H10": {0: 1.0, 1: None}, "F100_95": {0: 3.1, 1: 0.2}}
    calls = []

    def __init__(self):
        fe = self
        fe.init_args = None

        class Obj:
            def __init__(s, **kw):
                s.__dict__.update(kw)

        class Img(Obj):
            @staticmethod
            def cat(lst):
                return Obj(bands=lst, reduceRegions=lambda collection, reducer, scale: Obj(getInfo=lambda: fe._reduce(lst, collection)))

            def __init__(s, i):
                assert i.startswith(ch.EE_ASSET + "/"), i          # chemin complet exigé par Earth Engine
                s.i, s.name = i.rsplit("/", 1)[-1], None

            def select(s, b):
                return s

            def rename(s, n):
                s.name = n
                return s

        class Reducer:
            @staticmethod
            def first():
                return Obj(forEach=lambda names: names)

        self.Image = Img
        self.Reducer = Reducer
        self.Geometry = Obj(Point=lambda c: c)
        self.Feature = lambda g, p=None: Obj(g=g, p=p)
        self.FeatureCollection = lambda x: x if isinstance(x, list) else Obj(getInfo=lambda: {"features": [{"properties": r} for r in fe.IMAGES]})
        self.ImageCollection = lambda a: Obj(map=lambda f: None)
        self.Dictionary = Obj(fromLists=lambda *a: None)
        self.ServiceAccountCredentials = lambda email, key_data: ("creds", email)
        self.Initialize = self._init
        self.Geometry.Rectangle = lambda c: c
        Reducer.max = staticmethod(lambda: Obj(combine=lambda other, sharedInputs=True: None))
        Reducer.count = staticmethod(lambda: None)
        Img.bandNames = lambda s: Obj(getInfo=lambda: ["inundation_depth"])
        Img.projection = lambda s: Obj(nominalScale=lambda: Obj(getInfo=lambda: 927.0))
        Img.reduceRegion = lambda s, reducer, geometry, scale, maxPixels: Obj(
            getInfo=lambda: {"inundation_depth_max": 1.2, "inundation_depth_count": 40})

    def _init(self, creds, project=None):
        self.init_args = (creds, project)

    def _reduce(self, lst, coll):
        res = []
        for f in coll:
            k = f.p["k"]
            pr = {"k": k}
            for im in lst:
                v = self.VALUES[im.i].get(k)
                if v is not None:
                    pr[im.name] = v
            res.append({"properties": pr})
        return {"features": res}


ACC = {"client_email": "svc@p.iam.gserviceaccount.com", "private_key": "-----BEGIN PRIVATE KEY-----\\nABC\\n-----END PRIVATE KEY-----\\n", "project_id": "proj"}


class TestEarthEngine(unittest.TestCase):
    def setUp(self):
        self.fe = FakeEE()
        from unittest import mock
        p = mock.patch.object(ch, "_ee", return_value=self.fe)
        p.start()
        self.addCleanup(p.stop)
        ch._EE_STATE.clear()
        ch._EE_IMAGES.clear()
        ch._EE_IMAGES["list"] = FakeEE.IMAGES                  # liste de la collection : la requête serveur est simulée ailleurs

    def test_asset_path(self):
        self.assertEqual(ch.asset_path("inuncoast_x"), "WRI/Aqueduct_Flood_Hazard_Maps/V2/inuncoast_x")
        self.assertEqual(ch.asset_path("A/B/c"), "A/B/c")

    def test_parse_account(self):
        self.assertIn("\n", ch.parse_account(ACC)["private_key"])
        with self.assertRaises(ch.TileError):
            ch.parse_account(None)
        with self.assertRaises(ch.TileError):
            ch.parse_account({"client_email": "a", "private_key": "pas une clé"})
        with self.assertRaises(ch.TileError):
            ch.parse_account("{pas json")

    def test_pick_image_ignores_types_and_floodtype(self):
        rows = FakeEE.IMAGES
        self.assertEqual(ch.pick_image(rows, "historical", "nosub", "hist", 100)["img_id"], "H100")
        self.assertEqual(ch.pick_image(rows, "historical", "nosub", "hist", 10)["img_id"], "H10")      # "10" texte = 10 nombre
        self.assertEqual(ch.pick_image(rows, "rcp8p5", "wtsub", "2050", 100, "95")["img_id"], "F100_95")
        self.assertIsNone(ch.pick_image(rows, "rcp8p5", "wtsub", "2050", 100, "5"))
        self.assertIsNone(ch.pick_image(rows, "historical", "wtsub", "hist", 100))

    def test_estimate(self):
        df = pd.DataFrame({"lat": [45.0, 46.0, "x"], "lon": [12.0, 13.0, 1]})
        out, s = ch.estimate_depths_ee(df, "lat", "lon", [10, 100, 500], ee_project="proj", ee_account=ACC)
        self.assertEqual(self.fe.init_args[1], "proj")
        self.assertEqual(out["submersion_RP100_raw"].iloc[0], 2.5)
        self.assertEqual(out["submersion_RP10_raw"].iloc[0], 1.0)
        self.assertEqual(out["submersion_RP100_raw"].iloc[1], 0.0)             # masqué = aucune submersion modélisée
        self.assertTrue(pd.isna(out["submersion_RP100_raw"].iloc[2]))          # coordonnées invalides : rien d'inventé
        self.assertIn("RP500", s["problemes"])                                  # absente de la collection : signalé
        self.assertIn("valeurs présentes", s["problemes"]["RP500"])
        self.assertEqual(out["submersion_statut"].iloc[2], "coordonnées invalides")
        self.assertTrue(out["submersion_statut"].iloc[0].startswith("partiel"))
        self.assertEqual(s["points_positifs"]["RP100"], 1)
        self.assertEqual(s["source"], "Earth Engine")

    def test_future_with_projection(self):
        df = pd.DataFrame({"lat": [45.0, 46.0], "lon": [12.0, 13.0]})
        out, s = ch.estimate_depths_ee(df, "lat", "lon", [100], "rcp8p5", "wtsub", "2050", "95", "proj", ACC)
        self.assertEqual(list(out["submersion_RP100_raw"]), [3.1, 0.2])
        self.assertEqual(s["problemes"], {})

    def test_init_failure_message(self):
        def boom(creds, project=None):
            raise RuntimeError("Caller does not have required permission")
        self.fe.Initialize = boom
        with self.assertRaises(ch.TileError) as cm:
            ch.init_ee("proj", ACC)
        self.assertIn("Resource Writer", str(cm.exception))

    def test_diagnose(self):
        rows = ch.diagnose_ee("proj", ACC)
        self.assertEqual([r["étape"] for r in rows][:5], ["secrets", "authentification", "collection", "image", "points de contrôle"])
        self.assertIn("image : bandes", [r["étape"] for r in rows])
        self.assertIn("max 1.2 m sur 40 px", rows[-1]["détail"])
        self.assertTrue(all(r["résultat"] == "OK" for r in rows))
        self.assertEqual(ch.diagnose_ee("proj", None)[0]["résultat"], "ÉCHEC")
        self.assertNotIn("PRIVATE KEY", str(rows))                             # la clé n'apparaît jamais dans le diagnostic
