import unittest

from catmap import activity as act
from catmap.area import AreaOptions, AreaQuery
from catmap.config import Settings
from catmap.countries import get_country_index
from catmap.naf import NafClassifier
from catmap.overpass import FetchResult
from tests.helpers import FakeResponse

CI = get_country_index()
NAF = NafClassifier()
EXAMPLES = [
    ("Trouver les fabricants de composants électroniques au Vietnam", "electronic_components", "VN"),
    ("Identifier les producteurs de matériaux de construction en Allemagne", "building_materials", "DE"),
    ("Rechercher les entreprises agroalimentaires au Brésil", "food_processing", "BR"),
    ("Identifier les entrepôts frigorifiques en Espagne", "cold_storage", "ES"),
    ("Trouver les entreprises spécialisées dans la fabrication de batteries en Chine", "batteries", "CN"),
]


class Client:
    def __init__(self, data):
        self.data, self.queries = data, []

    def fetch(self, query, on_event=None):
        self.queries.append(query)
        return FetchResult(self.data, "http://x", "2026-01-01T00:00:00Z", False, 1, 0.1)


class T(unittest.TestCase):
    def test_examples_from_request(self):
        for text, aid, iso in EXAMPLES:
            i = act.interpret_keywords(text, CI)
            self.assertIn(aid, i.criteria.activities, text)
            self.assertEqual(i.country_iso2, iso, text)

    def test_normalize_naf(self):
        self.assertEqual(act.normalize_divisions(["26.11Z", "2611Z", "10", "99", "zz"]), ("26", "10"))
        self.assertIn("10", act.normalize_divisions(["C"]))
        self.assertNotIn("47", act.normalize_divisions(["C"]))

    def test_query_country_and_circle(self):
        s = Settings()
        q = act.build_activity_query(act.ActivityScope("country", iso2="ES", country="Espagne"),
                                     act.Criteria(("cold_storage",)), s, NAF)
        self.assertIn('area["ISO3166-1"="ES"]', q)
        self.assertIn("out center tags", q)
        q = act.build_activity_query(act.ActivityScope("circle", area=AreaQuery(48.8, 2.3, 5000)),
                                     act.Criteria(naf_divisions=("56",)), s, NAF)
        self.assertIn("around:5000", q)
        self.assertIn('"amenity"="restaurant"', q)
        with self.assertRaises(ValueError):
            act.build_activity_query(act.ActivityScope("circle", area=AreaQuery(48.8, 2.3, 50000)), act.Criteria(("batteries",)), s)
        with self.assertRaises(ValueError):
            act.build_activity_query(act.ActivityScope("country", iso2="E'"), act.Criteria(("batteries",)), s)
        with self.assertRaises(ValueError):
            act.build_activity_query(act.ActivityScope("country", iso2="ES"), act.Criteria(), s)

    def test_locate_country(self):
        data = {"elements": [
            {"type": "node", "id": 1, "lat": 40.4, "lon": -3.7, "tags": {"name": "Frigorificos Madrid", "industrial": "warehouse"}},
            {"type": "way", "id": 2, "center": {"lat": 41.4, "lon": 2.17}, "tags": {"name": "Cold Logistics SA", "building": "warehouse"}},
            {"type": "node", "id": 3, "lat": 40.0, "lon": -3.0, "tags": {"name": "Cafe del frío", "amenity": "cafe"}},
            {"type": "node", "id": 4, "lat": 40.1, "lon": -3.1, "tags": {"industrial": "warehouse"}},
        ]}
        c = Client(data)
        res = act.locate_activity(c, Settings(), act.ActivityScope("country", iso2="ES", country="Espagne"),
                                  act.Criteria(("cold_storage",), label="x"), AreaOptions(), CI, NAF)
        self.assertEqual(res.report.iloc[0]["statut"], "ok")
        self.assertEqual(sorted(res.sites["name"]), ["Cold Logistics SA", "Frigorificos Madrid"])
        self.assertTrue(res.sites["activite_critere"].str.startswith("nom~").all())
        self.assertEqual(set(res.sites["naf_division_code"]), {"52"})   # division reprise de l'activité (tag industrial=* générique)
        self.assertNotIn("distance_centre_m", res.sites.columns)
        self.assertEqual(res.meta["area"]["iso2"], "ES")

    def test_locate_circle_and_naf_tags(self):
        data = {"elements": [
            {"type": "node", "id": 1, "lat": 48.8566, "lon": 2.3522, "tags": {"name": "Chez A", "amenity": "restaurant"}},
            {"type": "node", "id": 2, "lat": 49.5, "lon": 2.3, "tags": {"name": "Loin", "amenity": "restaurant"}},
        ]}
        res = act.locate_activity(Client(data), Settings(), act.ActivityScope("circle", area=AreaQuery(48.8566, 2.3522, 1000)),
                                  act.Criteria(naf_divisions=("56",)), AreaOptions(), CI, NAF)
        self.assertEqual(list(res.sites["name"]), ["Chez A"])
        self.assertIn("distance_centre_m", res.sites.columns)
        self.assertEqual(res.sites.iloc[0]["naf_division_code"], "56")

    def test_no_criteria_is_error_not_exception(self):
        res = act.locate_activity(Client({}), Settings(), act.ActivityScope("country", iso2="ES"), act.Criteria(), AreaOptions())
        self.assertEqual(res.report.iloc[0]["statut"], "erreur")

    def test_ai_output_is_whitelisted(self):
        i = act.validate_ai_output({
            "activity_ids": ["batteries", "nope"], "naf_divisions": ["27", "ab", "99"],
            "osm_tags": [["industrial", "factory"], ["evil", "x"], ["craft", 'a"];out;']],
            "name_keywords": ["battery", 'x"];drop', "pin"], "country_iso2": "vn", "summary": "Batteries"})
        c = i.criteria
        self.assertEqual(c.activities, ("batteries",))
        self.assertEqual(c.naf_divisions, ("27",))
        self.assertEqual(c.extra_tags, (("industrial", "factory"),))
        self.assertEqual(c.extra_names, ("battery", "pin"))
        self.assertEqual(i.country_iso2, "VN")

    def test_interpret_ai_linkup(self):
        class S:
            def __init__(self, r): self.r, self.kw = r, None
            def post(self, url, **kw):
                self.kw = kw
                return self.r
        ok = FakeResponse(200, {"activity_ids": ["batteries"], "naf_divisions": ["27"], "osm_tags": [], "name_keywords": ["battery"],
                                "country_iso2": "CN"})
        s = S(ok)
        i = act.interpret_ai("batteries en Chine", "KEY", session=s)
        self.assertEqual(i.country_iso2, "CN")
        self.assertEqual(i.source, act.AI_SOURCE)
        self.assertEqual(s.kw["json"]["outputType"], "structured")
        self.assertEqual(s.kw["headers"]["Authorization"], "Bearer KEY")      # clé en en-tête uniquement
        wrapped = FakeResponse(200, {"data": {"activity_ids": ["metals"], "naf_divisions": []}})
        self.assertEqual(act.interpret_ai("x", "KEY", session=S(wrapped)).criteria.activities, ("metals",))
        with self.assertRaises(RuntimeError):
            act.interpret_ai("x", "KEY", session=S(FakeResponse(401, text="no")))
        with self.assertRaises(RuntimeError):
            act.interpret_ai("x", "KEY", session=S(FakeResponse(200, {"foo": 1})))
        with self.assertRaises(RuntimeError):
            act.interpret_ai("x", "")

    def test_lookup_term_table_first(self):
        c = act.lookup_term("café", NAF).criteria
        self.assertEqual(c.extra_tags, (("amenity", "cafe"),))
        self.assertEqual(act.lookup_term("Boulangerie", NAF).criteria.extra_tags, (("shop", "bakery"),))
        self.assertEqual(act.lookup_term("pharmacie", NAF).criteria.activities, ())      # tag exact prioritaire
        self.assertIn("47", act.lookup_term("commerce", NAF).criteria.naf_divisions)
        self.assertEqual(act.lookup_term("restauration", NAF).criteria.naf_divisions, ("56", "55"))
        self.assertEqual(act.lookup_term("composants électroniques", NAF).criteria.activities, ("electronic_components",))
        self.assertTrue(act.lookup_term("zzzzqq", NAF).criteria.is_empty())     # -> l'IA n'est appelée qu'ici
        self.assertTrue(act.lookup_term("  ", NAF).criteria.is_empty())

    def test_merge(self):
        kw = act.interpret_keywords("batteries", CI)
        ai = act.validate_ai_output({"activity_ids": ["metals"], "country_iso2": "CN"})
        m = act.merge_interpretations(kw, ai, CI)
        self.assertEqual(set(m.criteria.activities), {"batteries", "metals"})
        self.assertEqual(m.country_iso2, "CN")


if __name__ == "__main__":
    unittest.main()
