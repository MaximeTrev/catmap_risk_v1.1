import tempfile
import unittest
from pathlib import Path

import pandas as pd
import requests

from catmap.analysis import completeness, summary_by_company, top_with_other
from catmap.countries import CountryIndex
from catmap.models import NameQuery, SearchOptions
from catmap.naf import NafClassifier
from catmap.overpass import OverpassClient
from catmap.pipeline import SITE_COLUMNS, geolocate
from tests.helpers import FakeResponse, FakeSession, node, payload, square, test_settings, way




class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ci = CountryIndex(); cls.naf = NafClassifier()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.s = test_settings(self.tmp, max_attempts=1)

    def run_pipeline(self, queries, responses):
        # une réponse par appel, dans l'ordre des entreprises
        sess = FakeSession(responses)
        client = OverpassClient(self.s, session=sess, sleep=lambda x: None)
        return geolocate(queries, SearchOptions(), client, self.s, self.ci, self.naf)

    def test_end_to_end_with_failures(self):
        ikea = payload([way(1, "IKEA Lyon", square(4.85, 45.75, 0.002), shop="furniture"),
                        node(2, "IKEA Berlin", 13.4, 52.5, shop="furniture"),
                        node(3, "Cafe IKEA", 139.69, 35.69, amenity="cafe")])
        r = self.run_pipeline(
            [NameQuery.make("IKEA SA"), NameQuery.make("Inconnu Corp"), NameQuery.make("Panne SAS")],
            [FakeResponse(200, ikea), FakeResponse(200, payload([])), requests.ConnectionError("down")])
        rep = r.report.set_index("nom_demande")
        self.assertEqual(rep.loc["IKEA SA", "statut"], "ok")
        self.assertEqual(rep.loc["IKEA SA", "nom_recherche"], "IKEA")            # nettoyage correct
        self.assertEqual(rep.loc["Inconnu Corp", "statut"], "aucun résultat")    # A6
        self.assertEqual(rep.loc["Panne SAS", "statut"], "erreur")                # un échec n'arrête pas le lot
        self.assertEqual(rep.loc["IKEA SA", "snapshot"], "2026-09-30T20:00:00Z")
        s = r.sites
        self.assertEqual(len(s), 3)
        self.assertEqual(set(s.pays), {"France", "Allemagne", "Japon"})
        self.assertEqual(s.set_index("osm_id").loc[3, "naf_section_code"], "I")
        self.assertEqual(s.set_index("osm_id").loc[1, "naf_section_code"], "G")
        self.assertTrue(set(SITE_COLUMNS) <= set(s.columns))
        summ = summary_by_company(s, r.report).set_index("nom_demande")
        self.assertEqual(summ.loc["IKEA SA", "n_sites"], 3); self.assertEqual(summ.loc["Panne SAS", "n_sites"], 0)
        self.assertEqual(completeness(s)["pays"], 100.0)

    def test_empty_result_is_safe(self):                                        # A6 : plus de KeyError
        r = self.run_pipeline([NameQuery.make("Rien")], [FakeResponse(200, payload([]))])
        self.assertEqual(len(r.sites), 0); self.assertTrue(set(SITE_COLUMNS) <= set(r.sites.columns))
        self.assertEqual(completeness(r.sites)["pays"], 0.0)
        self.assertEqual(summary_by_company(r.sites, r.report).loc[0, "n_sites"], 0)

    def test_partial_result_kept_but_flagged(self):                             # A4
        bad = payload([node(1, "IKEA", 2.35, 48.85, shop="furniture")], remark="runtime error: Query timed out")
        r = self.run_pipeline([NameQuery.make("IKEA")], [FakeResponse(200, bad)])
        self.assertEqual(r.report.loc[0, "statut"], "partiel"); self.assertEqual(len(r.sites), 1)
        self.assertEqual(list(self.tmp.glob("*.json")), [])

    def test_second_run_uses_cache(self):                                       # A20
        good = payload([node(1, "IKEA", 2.35, 48.85)])
        sess = FakeSession([FakeResponse(200, good)])
        client = OverpassClient(self.s, session=sess, sleep=lambda x: None)
        geolocate([NameQuery.make("IKEA")], SearchOptions(), client, self.s)
        r2 = geolocate([NameQuery.make("IKEA")], SearchOptions(), client, self.s)
        self.assertEqual(len(sess.calls), 1); self.assertTrue(r2.report.loc[0, "depuis_cache"])

    def test_threads(self):
        s = self.s.with_overrides(max_workers=4)
        sess = FakeSession([FakeResponse(200, payload([node(1, "Alpha", 2.35, 48.85)]))])
        client = OverpassClient(s, session=sess, sleep=lambda x: None)
        qs = [NameQuery.make(n) for n in ("Alpha", "Alpha Bis", "Alpha Ter", "Alpha Quad")]
        done = []
        r = geolocate(qs, SearchOptions(), client, s, progress=lambda d, t, n: done.append((d, t)))
        self.assertEqual(len(r.report), 4); self.assertEqual(done[-1], (4, 4))

    def test_duplicate_requests_collapsed(self):
        r = self.run_pipeline([NameQuery.make("IKEA"), NameQuery.make("ikea")], [FakeResponse(200, payload([]))])
        self.assertEqual(len(r.report), 1)


class TestAnalysis(unittest.TestCase):
    def test_top_with_other(self):
        s = pd.Series(list("aaabbccddeeffgg"))
        t = top_with_other(s, top_n=3)
        self.assertEqual(list(t.modalite), ["a", "b", "c", "Autres"]); self.assertEqual(t.n.sum(), 15)
        self.assertTrue(top_with_other(pd.Series([], dtype=object)).empty)


if __name__ == "__main__":
    unittest.main()
