import unittest
from unittest import mock

import pandas as pd
import requests

from catmap import supply as sp
from catmap import websearch as ws


class R:
    def __init__(self, code=200, js=None):
        self.status_code, self._js = code, js
    def json(self):
        if self._js is None:
            raise ValueError
        return self._js


class T(unittest.TestCase):
    def test_search_dedupes_and_ranks(self):
        class S:
            def post(self, url, **k):
                return R(200, {"results": [{"type": "text", "name": "News", "url": "https://news.com/ikea", "content": "x"},
                                           {"type": "text", "name": "List", "url": "https://www.ikea.com/suppliers.pdf", "content": "y"}]})
        res = ws.search_documents(["a", "b"], "K", session=S())
        self.assertEqual(len(res), 2)                                  # doublons entre requêtes retirés
        self.assertEqual(ws.rank(res, "IKEA")[0]["url"], "https://www.ikea.com/suppliers.pdf")

    def test_errors_are_readable(self):
        for code, msg in ((401, "refusée"), (402, "Crédits"), (429, "Quota"), (500, "HTTP 500")):
            class S:
                def post(self, url, code=code, **k):
                    return R(code)
            with self.assertRaises(ws.SearchError) as c:
                ws.search_documents(["q"], "K", session=S())
            self.assertIn(msg, str(c.exception))

        class Bad:
            def post(self, url, **k):
                return R(200, {"oops": 1})
        with self.assertRaises(ws.SearchError):
            ws.search_documents(["q"], "K", session=Bad())

        class Down:
            def post(self, url, **k):
                raise requests.ConnectionError()
        with self.assertRaises(ws.SearchError):
            ws.search_documents(["q"], "K", session=Down())

    def test_analyse_urls_never_raises(self):
        html = ("<table><tr><th>Supplier</th><th>Country</th></tr>"
                + "".join(f"<tr><td>Factory {i} Ltd</td><td>PL</td></tr>" for i in range(6)) + "</table>")

        def fake_fetch(url, **k):
            if "bad" in url:
                raise sp.SupplyError("HTTP 404")
            return html.encode(), "text/html"
        with mock.patch.object(sp, "fetch_url", side_effect=fake_fetch):
            out = sp.analyse_urls(["https://a.com/x", "https://a.com/bad"])
        self.assertEqual(len(out[0]["tables"]), 1)
        self.assertEqual(out[1]["error"], "HTTP 404")

    def test_extract_suppliers_and_quote_verification(self):
        payload = {"suppliers": [
            {"name": "Acme Furniture", "country": "Poland", "relation": "fabricant", "tier": "1",
             "source_url": "https://a.com/report", "quote": "Acme Furniture manufactures our chairs"},
            {"name": "Ghost Ltd", "source_url": "https://a.com/report", "quote": "Ghost Ltd supplies everything"},
            {"name": ""}]}

        class S:
            def post(self, url, json=None, **k):
                assert json["outputType"] == "structured" and "suppliers" in json["structuredOutputSchema"]
                return R(200, payload)
        items = ws.extract_suppliers("IKEA", "K", session=S())
        self.assertEqual(len(items), 2)                                  # ligne sans nom écartée
        page = b"<html><body><p>In 2025, Acme Furniture manufactures our chairs in Poland.</p></body></html>"
        with mock.patch.object(sp, "fetch_url", return_value=(page, "text/html")):
            df = sp.candidates_from_ai(items)
        self.assertEqual(df["confiance"].tolist(), ["C", "D"])           # la citation inventée n'est pas validée
        self.assertIn("vérifiée", df.iloc[0]["source"])

    def test_extract_unexpected_format(self):
        class S:
            def post(self, url, **k):
                return R(200, {"foo": 1})
        with self.assertRaises(ws.SearchError):
            ws.extract_suppliers("IKEA", "K", session=S())

    def test_post_404_falls_back_to_get(self):
        calls = []

        class S:
            def post(self, url, **k):
                calls.append("post")
                return R(404)
            def get(self, url, **k):
                calls.append("get")
                return R(200, {"results": [{"name": "A", "url": "https://a.com/x", "content": ""}]})
        self.assertEqual(len(ws.search_documents(["q"], "K", session=S())), 1)
        self.assertEqual(calls, ["post", "get"])

    def test_useful_tables_filters_small(self):
        small = pd.DataFrame({"Supplier": ["A", "B"]})
        self.assertEqual(sp.useful_tables([("t", small)]), [])


if __name__ == "__main__":
    unittest.main()
