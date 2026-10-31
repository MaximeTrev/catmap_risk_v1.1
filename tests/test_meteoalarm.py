import unittest
from datetime import datetime, timezone

import numpy as np

from catmap import meteoalarm as ma

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)


def alert(level=3, typ=10, poly="48,2 48,3 49,3 49,2 48,2", expires="2026-10-28T12:00:00+00:00", **kw):
    info = {"language": "en-GB", "event": "Rain", "severity": "Severe", "urgency": "Future", "certainty": "Likely",
            "onset": "2026-10-27T10:00:00+00:00", "expires": expires, "headline": "Heavy rain",
            "parameter": [{"valueName": "awareness_level", "value": f"{level}; orange; Severe"},
                          {"valueName": "awareness_type", "value": f"{typ}; Rain"}],
            "area": [{"areaDesc": "Zone A", "polygon": [poly] if poly else [], "geocode": [{"valueName": "EMMA_ID", "value": "FR001"}]}]}
    info.update(kw)
    return {"alert": {"identifier": "x1", "sender": "s", "sent": "2026-10-27T09:00:00+00:00", "status": "Actual",
                      "msgType": "Alert", "info": [info]}}


class T(unittest.TestCase):
    def test_parse_basic(self):
        df, rings = ma.parse_feed({"warnings": [alert()]}, "france", now=NOW)
        self.assertEqual(len(df), 1)
        r = df.iloc[0]
        self.assertEqual((r.niveau, r.type, r.pays), ("Orange", "Fortes pluies", "France"))
        self.assertTrue(r.avec_contour)
        self.assertEqual(rings[0][0][0].tolist(), [2.0, 48.0])        # (lon, lat)

    def test_expired_and_cancel_filtered(self):
        old = alert(expires="2026-10-26T00:00:00+00:00")
        canc = alert(); canc["alert"]["msgType"] = "Cancel"
        df, _ = ma.parse_feed([old, canc], "france", now=NOW)
        self.assertTrue(df.empty)

    def test_languages_grouped(self):
        a = alert()
        a["alert"]["info"].append(dict(a["alert"]["info"][0], language="fr-FR", headline="Fortes pluies"))
        df, _ = ma.parse_feed([a], "france", now=NOW)
        self.assertEqual(len(df), 1)

    def test_no_polygon_and_severity_fallback(self):
        a = alert(poly=None, parameter=[])
        df, rings = ma.parse_feed([a], "austria", now=NOW)
        self.assertEqual(df.iloc[0].niveau, "Orange")
        self.assertFalse(df.iloc[0].avec_contour)
        self.assertEqual(rings, {})

    def test_unknown_format(self):
        with self.assertRaises(ma.MeteoAlarmError):
            ma.parse_feed({"foo": 1}, "france")

    def test_expose_points(self):
        df, rings = ma.parse_feed([alert(), alert(level=2, poly="0,0 0,1 1,1 1,0")], "france", now=NOW)
        res = ma.expose_points(np.array([48.5, 0.5, 60.0]), np.array([2.5, 0.5, 5.0]), df, rings)
        self.assertEqual(res["europe_niveau"].tolist(), ["Orange", "Jaune", None])
        self.assertIn("Fortes pluies", res.loc[0, "europe_types"])

    def test_fetch_all_errors(self):
        class R:
            def __init__(s, c, j=None): s.status_code, s._j = c, j
            def json(s): return s._j
        class S:
            def get(s, url, **k):
                return R(200, [alert()]) if "france" in url else R(404)
        df, rings, errs = ma.fetch_all(["france", "spain"], session=S(), now=NOW)
        self.assertEqual(len(df), 1)
        self.assertIn("Espagne", errs)
        self.assertEqual(len(ma.worst_by_country(df)), 1)


if __name__ == "__main__":
    unittest.main()
