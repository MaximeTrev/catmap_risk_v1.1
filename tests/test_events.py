import unittest

import pandas as pd
import requests

from catmap import events as ev
from tests.helpers import FakeResponse

ACT = {"count": 3, "next": None, "results": [
    {"code": "EMSR900", "name": "Flood in Spain", "category": {"slug": "flood", "name": "Flood"},
     "countries": [{"short_name": "Spain"}], "centroid": "SRID=4326;POINT (-0.4 39.5)",
     "activationTime": "2026-10-30T10:00:00Z", "lastUpdate": "2026-11-01T10:00:00Z", "closed": False,
     "drmPhase": "response", "n_products": 4},
    {"code": "EMSR901", "name": "Wildfire in Portugal", "category": {"slug": "wildfire", "name": "Wildfire"},
     "countries": [{"short_name": "Portugal"}], "centroid": "POINT (-8.0 40.0)",
     "activationTime": "2026-06-01T10:00:00Z", "closed": True},
    {"code": "EMSR902", "name": "Earthquake", "category": "Earthquake", "centroid": None,
     "activationTime": "2026-10-01T00:00:00Z"},
    {"name": "no code"}]}

CAPS = """<?xml version="1.0"?><WMS_Capabilities xmlns="http://www.opengis.net/wms"><Capability><Layer>
<Title>root</Title><Layer><Name>mf010.fwi</Name><Title>Fire Weather Index</Title></Layer>
<Layer><Name>nrt.hotspots</Name><Title>Active fire hotspots</Title></Layer>
<Layer><Name>countries</Name><Title>Borders</Title></Layer></Layer></Capability></WMS_Capabilities>"""


class Sess:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        for frag, resp in self.routes:
            if frag in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return FakeResponse(404)


class Activations(unittest.TestCase):
    def test_parse(self):
        df = ev.parse_activations(ACT["results"])
        self.assertEqual(list(df["code"]), ["EMSR900", "EMSR902", "EMSR901"])      # tri récent -> ancien
        r = df.set_index("code")
        self.assertEqual(r.loc["EMSR900", "genre"], "flood")
        self.assertEqual(r.loc["EMSR901", "genre"], "fire")
        self.assertEqual(r.loc["EMSR902", "genre"], "other")
        self.assertAlmostEqual(r.loc["EMSR900", "lat"], 39.5)
        self.assertAlmostEqual(r.loc["EMSR900", "long"], -0.4)
        self.assertTrue(pd.isna(r.loc["EMSR902", "lat"]))
        self.assertTrue(r.loc["EMSR900", "lien"].endswith("/EMSR900/"))

    def test_fetch_and_filter(self):
        s = Sess([("activations/api", FakeResponse(200, ACT))])
        df, diag = ev.fetch_activations(s)
        self.assertEqual(len(df), 3)
        self.assertEqual(diag[0]["statut"], 200)
        now = pd.Timestamp("2026-11-03", tz="UTC")
        self.assertEqual(list(ev.filter_activations(df, ["flood", "fire"], 30, now=now)["code"]), ["EMSR900"])
        self.assertEqual(len(ev.filter_activations(df, ["fire"], None, only_open=True, now=now)), 0)

    def test_pagination_follows_next(self):
        p1 = {"count": 2, "next": "https://mapping.emergency.copernicus.eu/activations/api/activations/?limit=1&offset=1",
              "results": [ACT["results"][0]]}
        p2 = {"count": 2, "next": None, "results": [ACT["results"][1]]}
        s = Sess([("offset=1", FakeResponse(200, p2)), ("activations/api", FakeResponse(200, p1))])
        df, _ = ev.fetch_activations(s)
        self.assertEqual(len(df), 2)

    def test_network_failure_is_reported_not_raised(self):
        df, diag = ev.fetch_activations(Sess([("activations", requests.ConnectionError("boom"))]))
        self.assertTrue(df.empty)
        self.assertEqual(diag[0]["statut"], "erreur")
        df, diag = ev.fetch_activations(Sess([]))
        self.assertTrue(df.empty)
        self.assertEqual(diag[0]["statut"], 404)


class Wms(unittest.TestCase):
    def test_capabilities(self):
        layers = ev.parse_capabilities(CAPS)
        self.assertEqual([n for n, _ in layers], ["mf010.fwi", "nrt.hotspots", "countries"])
        self.assertEqual(ev.pick_layers(layers, ev.FIRE_LAYER_WORDS), ["mf010.fwi", "nrt.hotspots"])
        self.assertEqual(ev.parse_capabilities("<html"), [])

    def test_discover_falls_back_to_second_host(self):
        s = Sess([("maps.effis", FakeResponse(503)), ("ies-ows", FakeResponse(200, text=CAPS))])
        url, layers, diag = ev.discover_wms(ev.EFFIS_WMS, s)
        self.assertIn("ies-ows", url)
        self.assertEqual(len(layers), 3)
        self.assertEqual(diag[0]["statut"], 503)

    def test_discover_none(self):
        url, layers, diag = ev.discover_wms(ev.EFAS_WMS, Sess([]))
        self.assertIsNone(url)
        self.assertEqual(len(diag), 2)


class Burnt(unittest.TestCase):
    GJ = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": 1, "area_ha": 120}, "geometry": {"type": "Point", "coordinates": [0, 0]}}]}

    def test_ok_and_table(self):
        data, diag = ev.burnt_areas(ev.BBOX_PRESETS["Europe"], Sess([("gwis", FakeResponse(200, self.GJ))]))
        self.assertEqual(len(data["features"]), 1)
        self.assertEqual(list(ev.burnt_table(data)["area_ha"]), [120])
        self.assertEqual(diag[-1]["statut"], 200)

    def test_non_json_is_diagnosed(self):
        data, diag = ev.burnt_areas(ev.BBOX_PRESETS["Europe"], Sess([("gwis", FakeResponse(200, text="<xml/>"))]))
        self.assertIsNone(data)
        self.assertTrue(any("non JSON" in d["remarque"] for d in diag))

    def test_truncation_warned(self):
        data, diag = ev.burnt_areas((0, 0, 1, 1), Sess([("gwis", FakeResponse(200, self.GJ))]), max_features=1)
        self.assertIn("tronquée", diag[-1]["remarque"])
        self.assertTrue(ev.burnt_table(None).empty)


if __name__ == "__main__":
    unittest.main()
