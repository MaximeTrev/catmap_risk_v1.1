import json
import unittest
from datetime import datetime, timedelta, timezone

from catmap import cyclones as cy
from catmap.cyclone_map import build_payload, icon_svg, popup_html, render_map_html
from tests.helpers import FakeResponse

ADV = int(datetime(2026, 10, 28, 12, tzinfo=timezone.utc).timestamp() * 1000)


def fc(tau, lat, lon, wind, gust=None, mslp=None, name="MELISSA", num=13, basin="AL", spd=8, dr=315):
    return {"attributes": {"STORMNAME": name, "STORMNUM": num, "BASIN": basin, "TAU": tau, "LAT": lat, "LON": lon,
                           "MAXWIND": wind, "GUST": gust, "MSLP": mslp, "TCDIR": dr, "TCSPD": spd,
                           "ADVDATE": ADV, "ADVISNUM": "12A", "TCDVLP": "Hurricane"}}


FORECAST = {"features": [
    fc(0, 16.0, -75.0, 90, 110, 975), fc(12, 16.8, -76.0, 105), fc(24, 17.9, -77.0, 120),
    fc(48, 20.0, -78.0, 135), fc(72, 23.0, -77.0, 60),
    fc(0, 14.0, 179.0, 40, 50, 998, name="TWENTY", num=3, basin="WP", spd=0), fc(24, 15.0, -178.0, 55, name="TWENTY", num=3, basin="WP")]}
OBSERVED = {"features": [
    {"attributes": {"STORMNAME": "MELISSA", "STORMNUM": 13, "BASIN": "AL", "DTG": ADV - 6 * 3600e3, "LAT": 15.5, "LON": -74.2, "INTENSITY": 80, "MSLP": 980}},
    {"attributes": {"STORMNAME": "MELISSA", "STORMNUM": 13, "BASIN": "AL", "DTG": ADV - 12 * 3600e3, "LAT": 15.0, "LON": -73.5, "INTENSITY": 70}},
    {"attributes": {"STORMNAME": "OLD", "STORMNUM": 9, "BASIN": "EP", "DTG": ADV - 100 * 3600e3, "LAT": 10, "LON": -120, "INTENSITY": 30}},
    {"attributes": {"STORMNAME": "FRESH", "STORMNUM": 5, "BASIN": "IO", "DTG": ADV - 3 * 3600e3, "LAT": -12, "LON": 70, "INTENSITY": 45}},
    {"attributes": {"STORMNAME": "FRESH", "STORMNUM": 5, "BASIN": "IO", "DTG": ADV - 9 * 3600e3, "LAT": -11.5, "LON": 71, "INTENSITY": 40}}]}


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


NOW = datetime(2026, 10, 28, 13, tzinfo=timezone.utc)


class Classification(unittest.TestCase):
    def test_saffir_simpson_thresholds(self):
        cases = {None: 0, 20: 0, 34: 0, 63: 0, 64: 1, 82: 1, 83: 2, 95: 2, 96: 3, 112: 3, 113: 4, 136: 4, 137: 5, 170: 5}
        for w, cat in cases.items():
            self.assertEqual(cy.category_from_wind(w), cat, w)

    def test_icon_keys_and_labels(self):
        self.assertEqual(cy.icon_key_for_wind(40), "TS")
        self.assertEqual(cy.icon_key_for_wind(120), "4")
        self.assertEqual(cy.stage_label(120, "AL"), "Ouragan catégorie 4")
        self.assertEqual(cy.stage_label(120, "WP"), "Typhon catégorie 4")
        self.assertEqual(cy.stage_label(120, "SH"), "Cyclone catégorie 4")
        self.assertEqual(cy.stage_label(45, "AL"), "Tempête tropicale")
        self.assertEqual(cy.stage_label(25, "AL"), "Dépression tropicale")

    def test_movement_and_units(self):
        self.assertEqual(cy.compass(315), "NW")
        self.assertIn("NW (315°) à 8 kt (15 km/h)", cy.movement_text(315, 8))
        self.assertEqual(cy.movement_text(None, None), "n/d")
        self.assertEqual(cy.movement_text(10, 0), "Quasi stationnaire")
        self.assertIn("167 km/h", cy.wind_text(90))
        self.assertEqual(cy.wind_text(None), "n/d")
        self.assertEqual(cy.pressure_text(975), "975 hPa")

    def test_bearing_speed(self):
        brg, spd = cy.bearing_speed((0, 0), (0, 1), 1)     # 1° vers l'est ≈ 60 nm en 1 h
        self.assertAlmostEqual(brg, 90, delta=0.5)
        self.assertAlmostEqual(spd, 60, delta=1)

    def test_unwrap(self):
        self.assertEqual(cy.unwrap_longitudes([179, -178, -170], 179), [179, 182, 190])


class Parsing(unittest.TestCase):
    def setUp(self):
        self.storms = cy.build_storms(cy.parse_forecast(FORECAST), cy.parse_observed(OBSERVED), now=NOW)
        self.by = {s.name: s for s in self.storms}

    def test_storms_found_and_sorted(self):
        self.assertEqual([s.name for s in self.storms], ["Melissa", "Fresh", "Twenty"])
        self.assertIn("Fresh", self.by)            # trace observée récente sans prévision
        self.assertNotIn("Old", self.by)           # trop ancienne

    def test_current_point_and_intensity_evolution(self):
        m = self.by["Melissa"]
        self.assertEqual(m.current.icon_key, "2")
        self.assertEqual([p.icon_key for p in m.forecast], ["2", "3", "4", "4", "TS"])
        self.assertEqual(m.current.gust_kt, 110)
        self.assertEqual(m.current.mslp_hpa, 975)
        self.assertEqual(len(m.past), 2)
        self.assertEqual(m.forecast[2].valid, datetime(2026, 10, 29, 12, tzinfo=timezone.utc))

    def test_antimeridian_unwrapped(self):
        t = self.by["Twenty"]
        self.assertEqual([round(p.lon) for p in t.forecast], [179, 182])
        self.assertEqual(t.kind, "typhon")

    def test_observed_only_fallback_movement(self):
        f = self.by["Fresh"]
        self.assertEqual(f.current.wind_kt, 45)
        self.assertIsNotNone(f.current.speed_kt)
        self.assertIsNone(f.current.gust_kt)

    def test_missing_forecast_wind_falls_back_to_observed(self):
        data = {"features": [fc(0, 16, -75, None), fc(12, 17, -76, 60)]}
        s = cy.build_storms(cy.parse_forecast(data), cy.parse_observed(OBSERVED), now=NOW)[0]
        self.assertEqual(s.current.wind_kt, 80)

    def test_invalid_rows_ignored(self):
        bad = {"features": [{"attributes": {"STORMNAME": "X", "TAU": 0}}, {"attributes": {"LAT": 1, "LON": 2}}]}
        self.assertEqual(cy.build_storms(cy.parse_forecast(bad), {}, now=NOW), [])


class Fetching(unittest.TestCase):
    def test_fetch_ok_and_query_params(self):
        s = Sess([("/0/query", FakeResponse(200, FORECAST)), ("/1/query", FakeResponse(200, OBSERVED))])
        storms = cy.fetch_storms(session=s, urls=["https://x/FeatureServer"], now=NOW)
        self.assertEqual(len(storms), 3)
        self.assertEqual(s.calls[0][1]["f"], "json")
        self.assertEqual(s.calls[0][1]["returnGeometry"], "false")

    def test_fallback_to_second_url(self):
        s = Sess([("bad", FakeResponse(500)), ("good/0/query", FakeResponse(200, FORECAST)),
                  ("good/1/query", FakeResponse(200, OBSERVED))])
        self.assertTrue(cy.fetch_storms(session=s, urls=["https://bad", "https://good"], now=NOW))

    def test_all_sources_down(self):
        with self.assertRaises(cy.CycloneError):
            cy.fetch_storms(session=Sess([("", FakeResponse(503))]), urls=["https://a"])

    def test_arcgis_error_payload(self):
        with self.assertRaises(cy.CycloneError):
            cy.fetch_storms(session=Sess([("", FakeResponse(200, {"error": {"message": "nope"}}))]), urls=["https://a"])

    def test_observed_layer_failure_tolerated(self):
        s = Sess([("/0/query", FakeResponse(200, FORECAST)), ("/1/query", FakeResponse(500))])
        self.assertEqual(len(cy.fetch_storms(session=s, urls=["https://x"], now=NOW)), 2)

    def test_wind_radii_optional(self):
        poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        gj = {"features": [{"geometry": poly, "properties": {"STORMNAME": "A"}}]}
        r = cy.fetch_wind_radii(session=Sess([("/query", FakeResponse(200, gj))]), urls=["https://x"])
        self.assertEqual(len(r["features"]), 3)
        self.assertEqual({f["properties"]["kt"] for f in r["features"]}, {34, 50, 64})
        self.assertEqual(cy.fetch_wind_radii(session=Sess([("", FakeResponse(500))]), urls=["https://x"]), {})


class Precipitation(unittest.TestCase):
    def test_attach_precipitation(self):
        storms = cy.build_storms(cy.parse_forecast(FORECAST), {}, now=NOW)
        m = [s for s in storms if s.name == "Melissa"][0]
        times = [(datetime(2026, 10, 28, 0) + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(0, 240)]
        res = [{"hourly": {"time": times, "precipitation": [1.0] * 240}} for _ in m.forecast]
        sess = Sess([("open-meteo", FakeResponse(200, res))])
        only = cy.Storm(key="x", name="M", basin="AL", current=m.current, forecast=m.forecast)
        self.assertTrue(cy.attach_precipitation([only], session=sess))
        self.assertTrue(all(p.precip_mm == 24.0 for p in m.forecast))
        params = sess.calls[0][1]
        self.assertEqual(len(params["latitude"].split(",")), 5)

    def test_service_down_is_tolerated(self):
        storms = cy.build_storms(cy.parse_forecast(FORECAST), {}, now=NOW)
        self.assertFalse(cy.attach_precipitation(storms, session=Sess([("", FakeResponse(500))])))
        self.assertTrue(all(p.precip_mm is None for s in storms for p in s.forecast))


class MapOutput(unittest.TestCase):
    def setUp(self):
        self.storms = cy.build_storms(cy.parse_forecast(FORECAST), cy.parse_observed(OBSERVED), now=NOW)

    def test_six_icons(self):
        svgs = {k: icon_svg(k) for k in cy.ICON_KEYS}
        self.assertEqual(len(svgs), 6)
        self.assertNotIn("<text", svgs["TS"])
        for k in "12345":
            self.assertIn(f">{k}</text>", svgs[k])
        with self.assertRaises(ValueError):
            icon_svg("6")

    def test_popup_contains_required_fields_and_escapes(self):
        s = self.storms[0]
        s.name = "<b>X</b>"
        h = popup_html(s, s.current)
        for label in ("Maximum Sustained Winds", "Max Wind Gusts", "Pressure", "Movement"):
            self.assertIn(label, h)
        self.assertNotIn("<b>X</b>", h)
        self.assertIn("&lt;b&gt;X", h)
        self.assertIn("110 kt", h)

    def test_payload_layers_and_icon_change(self):
        pl = build_payload(self.storms, {})
        self.assertEqual(len(pl["storms"]), 3)
        icons = [p["icon"] for p in pl["forecast_points"]][:4]
        self.assertEqual(icons, ["3", "4", "4", "TS"])        # le symbole évolue le long de la prévision
        mel = pl["storms"][0]
        self.assertIn("Melissa", mel["html"])                  # nom au-dessus du symbole
        self.assertLess(mel["html"].index("Melissa"), mel["html"].index("<svg"))
        colors = [s["color"] for s in pl["segments"]][:4]
        self.assertEqual(len(set(colors)), 3)

    def test_html_is_valid_json_payload(self):
        h = render_map_html(build_payload(self.storms, {}))
        self.assertIn("L.control.layers", h)
        self.assertEqual(h.count("</script>"), 2)

    def test_empty(self):
        h = render_map_html(build_payload([], {}))
        self.assertIn("Aucun cyclone", h)


if __name__ == "__main__":
    unittest.main()
