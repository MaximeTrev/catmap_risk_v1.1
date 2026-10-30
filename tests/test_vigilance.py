import unittest
import pandas as pd
from catmap.config import Settings, load_settings
from catmap.vigilance import (DepartementIndex, MeteoFranceAuth, VigilanceError, VigilanceFormatError,
                              expose_sites, fetch_vigilance, parse_vigilance, worst_by_domain)
from tests.helpers import FakeResponse

PAYLOAD = {"update_time": "2026-10-01T06:00:00Z", "product": {"periods": [
    {"echeance": "J", "begin_validity_time": "2026-10-01T06:00:00Z", "end_validity_time": "2026-10-02T06:00:00Z",
     "timelaps": {"domain_ids": [
        {"domain_id": "69", "max_color_id": 3, "phenomenon_items": [
            {"phenomenon_id": "2", "phenomenon_max_color_id": 3}, {"phenomenon_id": "3", "phenomenon_max_color_id": 2},
            {"phenomenon_id": "1", "phenomenon_max_color_id": 1}]},
        {"domain_id": "75", "max_color_id": 1, "phenomenon_items": [{"phenomenon_id": "1", "phenomenon_max_color_id": 1}]},
        {"domain_id": "2A", "max_color_id": 4, "phenomenon_items": [{"phenomenon_id": "1", "phenomenon_max_color_id": 4}]}]}}]}}

def sq(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]

GEO = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"code": "69", "nom": "Rhône"}, "geometry": {"type": "Polygon", "coordinates": sq(4, 45, 5, 46)}},
    {"type": "Feature", "properties": {"code": "75", "nom": "Paris"}, "geometry": {"type": "Polygon", "coordinates": sq(2, 48.7, 2.6, 49)}},
    {"type": "Feature", "properties": {"code": "2A", "nom": "Corse-du-Sud"}, "geometry": {"type": "MultiPolygon", "coordinates": [sq(8.5, 41.5, 9.2, 42.2)]}}]}


class TestVigilance(unittest.TestCase):
    def test_parse_real_structure(self):
        df = parse_vigilance(PAYLOAD)
        self.assertEqual(len(df), 5)
        self.assertEqual(set(df.columns), {"periode", "debut", "fin", "domaine", "phenomene_id", "phenomene", "couleur_id", "couleur"})
        w = worst_by_domain(df).set_index("domaine")
        self.assertEqual(w.loc["69", "couleur"], "Orange")
        self.assertIn("Pluie-inondation (Orange)", w.loc["69", "phenomenes"])
        self.assertEqual(w.loc["75", "phenomenes"], "")

    def test_parse_geojson_fallback_and_unknown(self):
        gj = {"type": "FeatureCollection", "features": [{"properties": {"code": "13", "couleur": "Jaune", "phenomene": "Orages"}}]}
        self.assertEqual(parse_vigilance(gj).loc[0, "couleur"], "Jaune")
        with self.assertRaises(VigilanceFormatError) as cm:
            parse_vigilance({"foo": 1, "bar": 2})
        self.assertIn("bar", str(cm.exception))

    def test_exposure(self):
        deps = DepartementIndex(GEO)
        sites = pd.DataFrame({"long": [4.85, 2.35, 9.0, 13.4, 4.5], "lat": [45.75, 48.85, 41.9, 52.5, 45.5],
                              "pays_iso2": ["FR", "FR", "FR", "DE", "FR"]})
        r = expose_sites(sites, parse_vigilance(PAYLOAD), deps)
        self.assertEqual(list(r.departement_code[:3]), ["69", "75", "2A"])
        self.assertEqual(list(r.vigilance_couleur[:3]), ["Orange", "Vert", "Rouge"])
        self.assertTrue(pd.isna(r.departement_code[3]))              # hors France : non traité
        self.assertEqual(r.vigilance_couleur_id[4], 3)

    def test_exposure_empty_and_no_france(self):
        deps = DepartementIndex(GEO)
        self.assertTrue(expose_sites(pd.DataFrame({"long": [], "lat": [], "pays_iso2": []}), parse_vigilance(PAYLOAD), deps).empty)

    def test_auth_errors_and_apikey(self):
        s = Settings()
        with self.assertRaises(VigilanceError):
            MeteoFranceAuth(s).headers()
        self.assertEqual(MeteoFranceAuth(Settings(mf_api_key="K")).headers()["apikey"], "K")

    def test_oauth_token_cached(self):
        calls = []
        class S:
            def post(self, url, data=None, auth=None, timeout=None):
                calls.append(url); r = FakeResponse(200, {"access_token": "T", "expires_in": 3600}); r.raise_for_status = lambda: None; return r
            def get(self, url, headers=None, timeout=None):
                return FakeResponse(200, PAYLOAD)
        s = Settings(mf_client_id="i", mf_client_secret="s")
        a = MeteoFranceAuth(s, session=S())
        self.assertEqual(a.headers()["Authorization"], "Bearer T"); a.headers()
        self.assertEqual(len(calls), 1)
        self.assertIn("product", fetch_vigilance(a, s))

    def test_http_errors(self):
        class S:
            def get(self, *a, **k): return FakeResponse(403, text="no")
        s = Settings(mf_api_key="K")
        with self.assertRaises(VigilanceError):
            fetch_vigilance(MeteoFranceAuth(s, session=S()), s)


class TestConfig(unittest.TestCase):
    def test_env_and_secrets_priority(self):
        s = load_settings(env={"METEOFRANCE_API_KEY": "env"}, secrets={"METEOFRANCE_API_KEY": "sec", "CATMAP_CONTACT": "a@b.c"})
        self.assertEqual((s.mf_api_key, s.contact), ("env", "a@b.c"))
        self.assertIn("a@b.c", s.user_agent)

    def test_local_defaults(self):
        s = load_settings(env={"OVERPASS_URLS": "http://localhost:12345/api/interpreter", "CATMAP_CACHE_TTL_HOURS": "never"})
        self.assertEqual((s.min_interval_s, s.cache_ttl_hours, s.uses_public_overpass), (0.0, None, False))
        self.assertGreater(s.max_workers, 1)
        p = load_settings(env={})
        self.assertEqual((p.min_interval_s, p.max_workers, p.uses_public_overpass), (1.0, 1, True))

    def test_no_secret_in_source(self):
        import pathlib, re
        for f in pathlib.Path(__file__).resolve().parent.parent.rglob("*.py"):
            if "tests" in f.parts:
                continue
            txt = f.read_text(encoding="utf-8")
            self.assertNotRegex(txt, r"[A-Za-z0-9._-]+@(?!example|square-management\.com)[A-Za-z0-9.-]+\.[a-z]{2,}", f"e-mail codé en dur dans {f}")   # seule exception : l'adresse de contact choisie par l'utilisateur
            self.assertNotRegex(txt, r"(?i)client_secret\s*=\s*['\"][A-Za-z0-9_\-]{12,}['\"]")


if __name__ == "__main__":
    unittest.main()
