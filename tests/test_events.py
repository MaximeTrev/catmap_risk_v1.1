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


SQUARE = [[[2.0, 40.0], [2.0, 41.0], [3.0, 41.0], [3.0, 40.0], [2.0, 40.0]]]


def _gj():
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"initialdate": "2026-10-01 00:00:00", "finaldate": "2026-10-05 00:00:00", "area_ha": 100},
         "geometry": {"type": "Polygon", "coordinates": SQUARE}},
        {"type": "Feature", "properties": {"initialdate": "2026-11-01 00:00:00", "finaldate": "2026-11-03 00:00:00", "area_ha": 50},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[[10.0, 10.0], [10.0, 11.0], [11.0, 11.0], [10.0, 10.0]]]]}}]}


class FireValues(unittest.TestCase):
    def test_fwi_classes(self):
        self.assertEqual(ev.fwi_class(3), (1, "Très faible"))
        self.assertEqual(ev.fwi_class(21.3)[1], "Élevé")
        self.assertEqual(ev.fwi_class(80), (6, "Extrême"))
        self.assertEqual(ev.fwi_class("x"), (0, ""))

    def test_first_number(self):
        self.assertEqual(ev.first_number("Layer 'mf010.fwi'\n  value_0 = '23.4'"), 23.4)
        self.assertEqual(ev.first_number('{"p":{"value": 7.5}}'), 7.5)
        self.assertIsNone(ev.first_number("rien"))

    def test_feature_info_tries_formats_and_reports(self):
        s = Sess([("wms", FakeResponse(200, text="<ServiceExceptionReport><ServiceException>x</ServiceException>"))])
        txt, diag = ev.get_feature_info("http://wms.test", "mf010.fwi", 45, 5, 6, s)
        self.assertEqual(txt, "")
        self.assertEqual(len(diag), 4)
        s = Sess([("wms", FakeResponse(200, text="fwi = 12.5"))])
        txt, _ = ev.get_feature_info("http://wms.test", "mf010.fwi", 45, 5, 6, s)
        self.assertEqual(ev.first_number(txt), 12.5)
        p = s.calls[0][1]
        self.assertEqual(p["request"], "GetFeatureInfo")
        south, west, north, east = map(float, p["bbox"].split(","))
        self.assertTrue(south < 45 < north and west < 5 < east)

    def test_sample_layer_groups_points_by_cell(self):
        s = Sess([("wms", FakeResponse(200, text="value = 30"))])
        vals, info = ev.sample_layer("http://wms.test", "mf010.fwi", [45.001, 45.002, 10.0], [5.001, 5.002, 10.0], s)
        self.assertEqual(info["mailles"], 2)
        self.assertEqual(list(vals), [30.0, 30.0, 30.0])

    def test_xml_value_and_getmap(self):
        self.assertEqual(ev.first_number("<FeatureInfo><time>2026</time><value>12.5</value></FeatureInfo>"), 12.5)
        u = ev.getmap_url("http://wms.test/api/wms/", "mapserver:RepPoints_High", time_value="2026-11-01-00-00")
        self.assertIn("request=GetMap", u)
        self.assertIn("time=2026-11-01-00-00", u)
        self.assertIn("layers=mapserver%3ARepPoints_High", u)
        self.assertIn("Aléa", ev.layer_note("FloodHazard100y"))

    def test_presence_without_number_and_zero_value(self):
        self.assertTrue(ev.info_has_feature("Results for FeatureType 'x':\nname = Rhine"))
        self.assertFalse(ev.info_has_feature(""))
        self.assertFalse(ev.info_has_feature('{"type":"FeatureCollection","features":[]}'))
        self.assertTrue(ev.info_has_feature('{"features":[{"properties":{"name":"a"}}]}'))
        self.assertFalse(ev.info_has_feature("no features were found"))
        s = Sess([("wms", FakeResponse(200, text="name = Rhine"))])
        vals, info = ev.sample_layer("http://wms.test", "x", [45.0], [5.0], s)
        self.assertTrue(vals[0] != vals[0])                      # pas de valeur numérique
        self.assertTrue(info["presence"][0])
        s = Sess([("wms", FakeResponse(200, text="value = 0"))])
        vals, _ = ev.sample_layer("http://wms.test", "x", [45.0], [5.0], s)
        self.assertEqual(vals[0], 0.0)                           # 0 est une valeur valide

    def test_empty_answer_is_final_and_bbox_filters(self):
        s = Sess([("wms", FakeResponse(200, text=""))])
        txt, diag = ev.get_feature_info("http://wms.test", "x", 45, 5, 6, s)
        self.assertEqual(len(diag), 1)                           # pas de 2e essai sur une réponse vide valide
        s = Sess([("wms", FakeResponse(200, text="value = 3"))])
        vals, info = ev.sample_layer("http://wms.test", "x", [45.0, -10.0], [5.0, 120.0], s, bbox=(-25, 30, 45, 72))
        self.assertEqual(info["hors_emprise"], 1)
        self.assertEqual(info["mailles"], 1)
        self.assertEqual(vals[0], 3.0)

    def test_layer_bbox_in_details(self):
        caps = CAPS.replace('<Name>mf010.fwi</Name>', '<Name>mf010.fwi</Name><EX_GeographicBoundingBox><westBoundLongitude>-25</westBoundLongitude>'
                            '<eastBoundLongitude>45</eastBoundLongitude><southBoundLatitude>30</southBoundLatitude>'
                            '<northBoundLatitude>72</northBoundLatitude></EX_GeographicBoundingBox>')
        d = {x["name"]: x for x in ev.fetch_layer_details("http://wms.test", Sess([("wms", FakeResponse(200, text=caps))]))}
        self.assertEqual(d["mf010.fwi"]["bbox"], (-25.0, 30.0, 45.0, 72.0))

    def test_legend_url(self):
        u = ev.legend_url("http://wms.test/effis", "mf010.fwi")
        self.assertIn("GetLegendGraphic", u)
        self.assertIn("layer=mf010.fwi", u)

    def test_layer_note_and_details(self):
        self.assertIn("carré", ev.layer_note("modis.ba"))
        self.assertIn("polygones", ev.layer_note("modis.ba.poly"))
        self.assertIn("Forêt-Météo", ev.layer_note("mf010.fwi"))
        caps = CAPS.replace('<Name>nrt.hotspots</Name>', '<Name>nrt.hotspots</Name><Abstract>Foyers</Abstract><Dimension name="time" units="ISO8601">2026-10-01/2026-11-01</Dimension>')
        d = {x["name"]: x for x in ev.fetch_layer_details("http://wms.test", Sess([("wms", FakeResponse(200, text=caps))]))}
        self.assertEqual(d["nrt.hotspots"]["abstract"], "Foyers")
        self.assertIn("2026-11-01", d["nrt.hotspots"]["time"])


class FireDates(unittest.TestCase):
    NOW = pd.Timestamp("2026-11-04", tz="UTC")

    def test_detect_and_defaults(self):
        t = ev.burnt_table(_gj())
        cols = ev.detect_date_columns(t)
        self.assertEqual(cols, ["initialdate", "finaldate"])
        self.assertEqual(ev.default_date_columns(cols), ("initialdate", "finaldate"))

    def test_filters(self):
        gj = _gj()
        n = lambda g: len(g["features"])
        self.assertEqual(n(ev.filter_burnt(gj, "tout", "initialdate", "finaldate")), 2)
        self.assertEqual(n(ev.filter_burnt(gj, "en cours", "initialdate", "finaldate", active_days=3, now=self.NOW)), 1)
        self.assertEqual(n(ev.filter_burnt(gj, "en cours", "initialdate", "finaldate", active_days=0, now=self.NOW)), 0)
        self.assertEqual(n(ev.filter_burnt(gj, "date", "initialdate", "finaldate", d1=pd.Timestamp("2026-10-03"))), 1)
        self.assertEqual(n(ev.filter_burnt(gj, "date", "initialdate", "finaldate", d1=pd.Timestamp("2026-10-20"))), 0)
        self.assertEqual(n(ev.filter_burnt(gj, "période", "initialdate", "finaldate", d1=pd.Timestamp("2026-10-04"),
                                           d2=pd.Timestamp("2026-11-02"))), 2)

    def test_cross_points(self):
        df = pd.DataFrame({"name": ["dedans", "pres", "loin"], "lat": ["40.5", "40.5", "50"], "long": ["2.5", "3.1", "20"]})
        out = ev.cross_with_burnt(df, "lat", "long", _gj(), snap_km=20, date_cols=("initialdate", "finaldate"))
        self.assertEqual(list(out["incendie_statut"]), ["Dans une zone brûlée", "À moins de 20 km", "Non touché"])
        self.assertEqual(out.loc[0, "incendie_area_ha"], 100)
        self.assertTrue(out.loc[1, "incendie_distance_km"] > 0)
        empty = ev.cross_with_burnt(df, "lat", "long", None)
        self.assertEqual(set(empty["incendie_statut"]), {"Non touché"})


if __name__ == "__main__":
    unittest.main()
