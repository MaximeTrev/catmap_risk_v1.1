import tempfile
import unittest
from pathlib import Path

import requests

from catmap.models import NameQuery, SearchOptions
from catmap.overpass import (OverpassClient, OverpassPartialResult, OverpassQueryError,
                             OverpassUnavailable, build_query)
from tests.helpers import FakeResponse, FakeSession, payload, test_settings


class TestBuildQuery(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = test_settings(self.tmp)

    def test_default_query_shape(self):
        q = build_query(NameQuery.make("IKEA"), SearchOptions(), self.s)
        self.assertTrue(q.startswith("[out:json][timeout:180];"))
        self.assertIn('[!"highway"][!"place"][!"junction"][!"boundary"]', q)
        self.assertIn("rel[type=multipolygon]", q)
        self.assertIn('["brand"~', q)
        self.assertNotIn('["operator"~', q)
        self.assertTrue(q.rstrip().endswith("out body geom;"))
        self.assertNotIn("\\", q)                      # aucun échappement fragile (A5)

    def test_options(self):
        q = build_query(NameQuery.make("Total", "q154037"),
                        SearchOptions(include_operator=True, include_relations=False, countries=("fr", "BE"),
                                      require_business_tag=False), self.s)
        self.assertIn('["operator"~', q)
        self.assertIn('"brand:wikidata"="Q154037"', q)
        self.assertNotIn("rel[", q)
        self.assertIn('area["ISO3166-1"="BE"][admin_level=2];area["ISO3166-1"="FR"][admin_level=2];', q)
        self.assertIn("(area.searchArea)", q)
        self.assertNotIn("(shop|", q)

    def test_bbox_and_validation(self):
        q = build_query(NameQuery.make("X1"), SearchOptions(bbox=(43, 5, 44, 6)), self.s)
        self.assertIn("(43.0,5.0,44.0,6.0)", q)
        with self.assertRaises(ValueError):
            build_query(NameQuery.make("X1"), SearchOptions(countries=("FRA",)), self.s)
        with self.assertRaises(ValueError):
            build_query(NameQuery.make("X1"), SearchOptions(bbox=(50, 0, 40, 1)), self.s)

    def test_maxsize(self):
        s = self.s.with_overrides(maxsize_mb=256)
        self.assertIn("[maxsize:268435456]", build_query(NameQuery.make("X1"), SearchOptions(), s))


class TestClient(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.s = test_settings(self.tmp)
        self.sleeps = []

    def client(self, script, **kw):
        sess = FakeSession(script)
        return OverpassClient(self.s.with_overrides(**kw), session=sess, sleep=self.sleeps.append), sess

    def test_success_and_cache(self):
        c, sess = self.client([FakeResponse(200, payload([]))])
        r1 = c.fetch("Q1"); r2 = c.fetch("Q1")
        self.assertFalse(r1.from_cache); self.assertTrue(r2.from_cache)
        self.assertEqual(len(sess.calls), 1)                    # A20 : pas de requête rejouée
        self.assertEqual(r1.snapshot, "2026-09-30T20:00:00Z")
        self.assertIn("test@example.org", sess.calls[0]["headers"]["User-Agent"])

    def test_retry_on_429_then_success(self):
        c, sess = self.client([FakeResponse(429), FakeResponse(200, payload([]))])
        r = c.fetch("Q"); self.assertEqual(r.attempts, 2); self.assertEqual(len(sess.calls), 2)
        self.assertEqual(len(self.sleeps), 1)

    def test_retry_on_503_and_failover(self):
        # a: 503, 503 (2 essais) puis b: 200
        c, sess = self.client([FakeResponse(503), FakeResponse(503), FakeResponse(200, payload([]))])
        r = c.fetch("Q")
        self.assertEqual([x["url"] for x in sess.calls], ["http://a/api", "http://a/api", "http://b/api"])
        self.assertEqual(r.endpoint, "http://b/api")

    def test_partial_result_is_not_accepted_nor_cached(self):     # A4
        remark = 'runtime error: Query timed out in "query" at line 3 after 181 seconds.'
        c, sess = self.client([FakeResponse(200, payload([{"type": "node", "id": 1}], remark=remark))])
        with self.assertRaises(OverpassPartialResult) as cm:
            c.fetch("Q")
        self.assertIn("timed out", cm.exception.remark)
        self.assertEqual(len(sess.calls), 2)                    # les deux serveurs ont été essayés
        self.assertEqual(list(self.tmp.glob("*.json")), [])     # jamais en cache

    def test_partial_then_good_server(self):
        bad = FakeResponse(200, payload([], remark="runtime error: Query run out of memory"))
        c, _ = self.client([bad, FakeResponse(200, payload([]))])
        self.assertEqual(c.fetch("Q").endpoint, "http://b/api")

    def test_400_raises_immediately(self):
        c, sess = self.client([FakeResponse(400, text="parse error")])
        with self.assertRaises(OverpassQueryError):
            c.fetch("Q")
        self.assertEqual(len(sess.calls), 1)

    def test_all_down(self):
        c, _ = self.client([requests.ConnectionError("boom")])
        with self.assertRaises(OverpassUnavailable):
            c.fetch("Q")

    def test_timeout_retried(self):
        c, sess = self.client([requests.Timeout("t"), FakeResponse(200, payload([]))])
        self.assertEqual(c.fetch("Q").attempts, 2)

    def test_non_json_goes_to_next_server(self):
        c, _ = self.client([FakeResponse(200, None, text="<html>"), FakeResponse(200, payload([]))])
        self.assertEqual(c.fetch("Q").endpoint, "http://b/api")

    def test_cache_ttl_and_namespace(self):
        c, sess = self.client([FakeResponse(200, payload([]))], cache_ttl_hours=0.0)
        c.fetch("Q"); c.fetch("Q")
        self.assertEqual(len(sess.calls), 2)                    # TTL 0 : toujours expiré
        c2, sess2 = self.client([FakeResponse(200, payload([]))], cache_namespace="snap-A", cache_ttl_hours=None)
        c2.fetch("Q"); c2.fetch("Q")
        c3, sess3 = self.client([FakeResponse(200, payload([]))], cache_namespace="snap-B", cache_ttl_hours=None)
        c3.fetch("Q")
        self.assertEqual((len(sess2.calls), len(sess3.calls)), (1, 1))

    def test_corrupt_cache_ignored(self):
        c, sess = self.client([FakeResponse(200, payload([]))])
        (self.tmp / f"{c._cache_key('Q')}.json").write_text("{corrupt")
        c.fetch("Q"); self.assertEqual(len(sess.calls), 1)


if __name__ == "__main__":
    unittest.main()
