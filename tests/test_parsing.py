import unittest
from catmap.models import NameQuery, SearchOptions
from catmap.parsing import parse_elements
from tests.helpers import node, payload, square, way


def run(elements, q="IKEA", **opt):
    return parse_elements(payload(elements), NameQuery.make(q), SearchOptions(**opt))


class TestParsing(unittest.TestCase):
    def test_closed_way_has_area_and_inside_point(self):
        r = run([way(1, "IKEA Lyon", square(5, 45, 0.002), shop="furniture")])
        rec = r.records[0]
        self.assertEqual(rec["geom_kind"], "polygon")
        self.assertAlmostEqual(rec["surface_m2"], 0.002 * 111_000 * 0.002 * 78_700, delta=rec["surface_m2"] * 0.01)
        self.assertEqual(rec["shop"], "furniture"); self.assertIsNone(rec["amenity"])   # None, plus "N/A"
        self.assertEqual(rec["osm_url"], "https://www.openstreetmap.org/way/1")

    def test_open_way_has_no_fake_area(self):                         # A8
        r = run([way(2, "IKEA Depot", [(5, 45), (5.001, 45), (5.001, 45.001)])])
        self.assertEqual(r.records[0]["geom_kind"], "line"); self.assertIsNone(r.records[0]["surface_m2"])

    def test_missing_geometry_does_not_crash(self):                   # A9
        r = run([{"type": "way", "id": 3, "tags": {"name": "IKEA X"}}])
        self.assertEqual((len(r.records), r.stats.n_no_geometry), (0, 1))

    def test_strict_word_match(self):
        r = run([node(1, "IKEA", 1, 1), node(2, "xxIKEAxx", 2, 2), node(3, "Pineapple", 3, 3), node(4, "unknown", 4, 4)])
        self.assertEqual([x["osm_id"] for x in r.records], [1]); self.assertEqual(r.stats.n_rejected_name, 2)
        loose = run([node(2, "xxIKEAxx", 2, 2)], strict_word_match=False)
        self.assertEqual(loose.records[0]["match_score"], 0.30)

    def test_brand_tag_matches_when_name_differs(self):               # A16
        r = run([node(1, "Magasin du centre", 1, 1, brand="IKEA", **{"brand:wikidata": "Q54078"})])
        self.assertEqual((r.records[0]["match_field"], r.records[0]["brand_wikidata"]), ("brand", "Q54078"))

    def test_node_in_polygon_deduplicated_and_tags_merged(self):
        poly = way(1, "IKEA Lyon", square(5, 45, 0.002), building="retail")
        n_in = node(2, "IKEA Lyon", 5.001, 45.001, shop="furniture")
        n_far = node(3, "IKEA Lyon", 6, 46)
        r = run([n_in, poly, n_far])
        self.assertEqual(r.stats.n_merged, 1); self.assertEqual(len(r.records), 2)
        kept = next(x for x in r.records if x["osm_type"] == "way")
        self.assertEqual((kept["shop"], kept["n_fusionnes"]), ("furniture", 1))   # tag repris du node
        self.assertNotIn("_outer", kept)

    def test_dedup_disabled(self):
        r = run([node(1, "IKEA", 1, 1), node(2, "IKEA", 1.00001, 1.00001)], dedup_radius_m=0)
        self.assertEqual(len(r.records), 2)

    def test_min_score_filter(self):
        r = run([node(1, "IKEA", 1, 1), node(2, "Magasin IKEA", 2, 2)], min_score=0.8)
        self.assertEqual(len(r.records), 1)

    def test_relation_multipolygon_area_with_hole(self):
        o = [{"lat": 45, "lon": 5}, {"lat": 45, "lon": 5.002}, {"lat": 45.002, "lon": 5.002}, {"lat": 45.002, "lon": 5}, {"lat": 45, "lon": 5}]
        i = [{"lat": 45.0005, "lon": 5.0005}, {"lat": 45.0005, "lon": 5.001}, {"lat": 45.001, "lon": 5.001}, {"lat": 45.001, "lon": 5.0005}, {"lat": 45.0005, "lon": 5.0005}]
        rel = {"type": "relation", "id": 9, "tags": {"name": "IKEA Park", "type": "multipolygon", "shop": "mall"},
               "members": [{"type": "way", "role": "outer", "geometry": o}, {"type": "way", "role": "inner", "geometry": i}]}
        full = run([way(1, "IKEA Park", [(p["lon"], p["lat"]) for p in o])]).records[0]["surface_m2"]
        r = run([rel]).records[0]
        self.assertEqual(r["geom_kind"], "multipolygon"); self.assertLess(r["surface_m2"], full)
        self.assertGreater(r["surface_m2"], 0.7 * full)

    def test_incomplete_relation_flagged(self):
        o = [{"lat": 45, "lon": 5}, {"lat": 45, "lon": 5.002}, {"lat": 45.002, "lon": 5.002}]
        rel = {"type": "relation", "id": 9, "tags": {"name": "IKEA Park"}, "members": [{"type": "way", "role": "outer", "geometry": o}]}
        r = run([rel])
        self.assertEqual(len(r.records), 0); self.assertEqual(r.stats.n_no_geometry, 1)

    def test_truncation(self):
        els = [node(i, "IKEA", i * 0.1, 10) for i in range(1, 6)]
        r = parse_elements(payload(els), NameQuery.make("IKEA"), SearchOptions(dedup_radius_m=0), max_sites=3)
        self.assertEqual((len(r.records), r.stats.truncated), (3, True))


if __name__ == "__main__":
    unittest.main()
