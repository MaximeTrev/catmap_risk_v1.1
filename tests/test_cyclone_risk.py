import unittest
from datetime import datetime, timezone

import pandas as pd

from catmap import cyclones as cy
from catmap import cyclone_risk as cr
from catmap.inputs import InputError
from tests.test_cyclones import FORECAST, NOW, OBSERVED

STORMS = cy.build_storms(cy.parse_forecast(FORECAST), cy.parse_observed(OBSERVED), now=NOW)


def fake_precip(points):
    return [10.0 + i for i, _ in enumerate(points)]


def run(rows, radius=100, **kw):
    df = pd.DataFrame(rows, columns=["nom", "lat", "long"])
    return cr.assess_points(df, "lat", "long", STORMS, radius_km=radius, precip_fetcher=fake_precip, now=NOW, **kw)


class Assess(unittest.TestCase):
    def test_columns_and_off_track(self):
        r = run([("loin", 48.85, 2.35)])
        row = r.iloc[0]
        self.assertEqual(row["impact"], 0)
        for c in ("max_sustained_winds", "max_wind_gusts", "total_precipitation", "cyclone"):
            self.assertIsNone(row[c])
        self.assertEqual(list(r.columns[:3]), ["nom", "lat", "long"])   # colonnes d'origine conservées

    def test_current_position_gives_first_step(self):
        # Melissa actuelle (16.0, -75.0) : déjà dans le rayon → première échéance (12 h)
        row = run([("ici", 16.0, -75.0)]).iloc[0]
        self.assertEqual(row["impact"], 12)
        self.assertEqual(row["cyclone"], "Melissa")
        self.assertEqual(row["max_sustained_winds"], round(105 * cy.KT_TO_KMH))   # max(point 0 h = 90, point 12 h = 105)
        self.assertEqual(row["max_wind_gusts"], round(110 * cy.KT_TO_KMH))        # seule rafale connue : point 0 h
        self.assertEqual(row["total_precipitation"], 10.0)

    def test_later_point_gets_later_impact(self):
        row = run([("plus loin", 20.0, -78.0)]).iloc[0]       # point de prévision à +48 h
        self.assertEqual(row["impact"], 48)
        self.assertEqual(row["max_sustained_winds"], round(135 * cy.KT_TO_KMH))

    def test_between_forecast_steps_rounds_up(self):
        # milieu du segment 24 h → 48 h : entrée dans le rayon entre les deux → échéance 48 h
        row = run([("milieu", 18.95, -77.5)], radius=20).iloc[0]
        self.assertEqual(row["impact"], 48)

    def test_radius_matters(self):
        near = (16.0, -73.8)       # ≈ 128 km à l'est de la position actuelle
        self.assertEqual(run([("p",) + near], radius=50).iloc[0]["impact"], 0)
        self.assertGreater(run([("p",) + near], radius=200).iloc[0]["impact"], 0)

    def test_antimeridian(self):
        row = run([("dateline", 14.7, -179.0)], radius=40).iloc[0]    # Twenty est à 179°E puis 182° (= -178°)
        self.assertEqual(row["cyclone"], "Twenty")
        self.assertEqual(row["impact"], 24)       # atteint vers +16 h → première échéance ≥ : 24 h

    def test_invalid_coordinates_kept(self):
        df = pd.DataFrame({"nom": ["a", "b"], "lat": ["abc", ""], "long": ["1", "2"]})
        r = cr.assess_points(df, "lat", "long", STORMS, precip_fetcher=fake_precip, now=NOW)
        self.assertEqual(list(r["impact"]), [0, 0])

    def test_no_storms(self):
        df = pd.DataFrame({"lat": [1.0], "long": [2.0]})
        r = cr.assess_points(df, "lat", "long", [], now=NOW)
        self.assertEqual(r.iloc[0]["impact"], 0)

    def test_precip_failure_gives_none_but_keeps_impact(self):
        df = pd.DataFrame({"lat": [16.0], "long": [-75.0]})
        r = cr.assess_points(df, "lat", "long", STORMS, precip_fetcher=lambda pts: [None] * len(pts), now=NOW)
        self.assertEqual(r.iloc[0]["impact"], 12)
        self.assertIsNone(r.iloc[0]["total_precipitation"])

    def test_duplicate_points_single_precip_query(self):
        calls = []
        def fetch(pts):
            calls.append(len(pts)); return [5.0] * len(pts)
        df = pd.DataFrame({"lat": [16.0, 16.001], "long": [-75.0, -75.001]})
        r = cr.assess_points(df, "lat", "long", STORMS, precip_fetcher=fetch, now=NOW)
        self.assertEqual(calls, [1])
        self.assertEqual(list(r["total_precipitation"]), [5.0, 5.0])


class MapPoints(unittest.TestCase):
    def test_points_payload_before_and_after(self):
        from catmap.cyclone_map import build_points, impact_color, render_map_html, build_payload
        df = pd.DataFrame({"name": ["<i>A</i>", "B"], "lat": [16.0, 48.85], "long": [-75.0, 2.35]})
        pts, trunc = build_points(df, "lat", "long", STORMS)
        self.assertFalse(trunc)
        self.assertEqual({p["color"] for p in pts}, {"#1f6fb5"})
        self.assertNotIn("<i>A</i>", pts[0]["popup"])
        res = cr.assess_points(df, "lat", "long", STORMS, precip_fetcher=fake_precip, now=NOW)
        pts, _ = build_points(res, "lat", "long", STORMS, analysed=True)
        self.assertEqual(pts[0]["color"], "#d7191c")                  # impact 12 h
        self.assertEqual(pts[1]["color"], "#2e9e5b")                  # hors trajectoire
        self.assertIn("max_wind_gusts", pts[0]["popup"])
        self.assertIn("hors trajectoire", pts[1]["tip"])
        self.assertEqual([impact_color(x) for x in (0, 12, 36, 60, 96)], ["#2e9e5b", "#d7191c", "#fd8d3c", "#fecc5c", "#ffffb2"])
        self.assertIn("Vos points", render_map_html(build_payload(STORMS, {}, points=pts, points_analysed=True)))

    def test_cap_keeps_impacted_first_and_dateline(self):
        from catmap.cyclone_map import build_points
        df = pd.DataFrame({"name": list("abc"), "lat": [0, 14.7, 1], "long": [10, -179, 11], "impact": [0, 24, 0]})
        pts, trunc = build_points(df, "lat", "long", STORMS, analysed=True, cap=2)
        self.assertTrue(trunc)
        self.assertEqual(pts[0]["tip"], "b — impact 24 h")
        self.assertAlmostEqual(pts[0]["lon"], 181)                    # replacé près de la trajectoire > 180°


class Reading(unittest.TestCase):
    def test_semicolon_decimal_comma(self):
        df, la, lo = cr.read_points_csv("nom;lat;long\nA;48,85;2,35\n".encode("utf-8"))
        self.assertEqual((la, lo), ("lat", "long"))
        r = cr.assess_points(df, la, lo, [], now=NOW)
        self.assertEqual(len(r), 1)

    def test_aliases_and_comma_separator(self):
        _, la, lo = cr.read_points_csv(b"Name,Latitude,Longitude\nA,1.5,2.5\n")
        self.assertEqual((la, lo), ("Latitude", "Longitude"))

    def test_missing_columns(self):
        with self.assertRaises(InputError):
            cr.read_points_csv(b"a;b\n1;2\n")

    def test_reserved_columns_refused(self):
        with self.assertRaises(InputError):
            cr.read_points_csv(b"lat;long;impact\n1;2;3\n")

    def test_empty(self):
        with self.assertRaises(InputError):
            cr.read_points_csv(b"  \n")

    def test_reserved_columns_allowed_when_not_checked(self):
        df, _, _ = cr.read_points_csv(b"lat;long;impact\n1;2;3\n", reserved_check=False)
        self.assertEqual(len(df), 1)

    def test_excel_export_sheet_sites(self):
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            pd.DataFrame({"x": [1]}).to_excel(xw, sheet_name="rapport", index=False)
            pd.DataFrame({"name": ["A"], "lat": [48.85], "long": [2.35]}).to_excel(xw, sheet_name="sites", index=False)
        df, la, lo = cr.read_points_csv(buf.getvalue(), "catmap_sites.xlsx")
        self.assertEqual((la, lo, len(df)), ("lat", "long", 1))


class PrecipFetch(unittest.TestCase):
    def test_site_precipitation_sum_and_coverage(self):
        from datetime import timedelta
        from tests.helpers import FakeResponse
        from tests.test_cyclones import Sess
        t0 = datetime(2026, 10, 28, 0, tzinfo=timezone.utc)
        times = [(t0 + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(48)]
        res = [{"hourly": {"time": times, "precipitation": [0.5] * 48}}]
        s = Sess([("open-meteo", FakeResponse(200, res))])
        out = cr.site_precipitation([(10, 20, t0 + timedelta(hours=2), t0 + timedelta(hours=14))], session=s)
        self.assertEqual(out, [6.0])
        out2 = cr.site_precipitation([(10, 20, t0 + timedelta(hours=40), t0 + timedelta(hours=80))], session=s)
        self.assertEqual(out2, [None])       # couverture insuffisante
        out3 = cr.site_precipitation([(10, 20, t0, t0 + timedelta(hours=5))], session=Sess([("", FakeResponse(500))]))
        self.assertEqual(out3, [None])


if __name__ == "__main__":
    unittest.main()
