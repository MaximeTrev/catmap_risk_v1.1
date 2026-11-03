import unittest

import pandas as pd

from catmap import supply as sp


class T(unittest.TestCase):
    def test_html_table_and_columns(self):
        html = ("<html><body><table><tr><th>Supplier Name</th><th>Country</th><th>City</th></tr>"
                "<tr><td>Acme Furniture Sp. z o.o.</td><td>Poland</td><td>Lodz</td></tr>"
                "<tr><td>Beta Textiles Ltd</td><td>India</td><td>Tiruppur</td></tr></table></body></html>")
        tabs = sp.tables_from_bytes(html.encode(), "x.html", "text/html")
        self.assertEqual(len(tabs), 1)
        name, country = sp.guess_columns(tabs[0][1])
        self.assertEqual((name, country), ("Supplier Name", "Country"))
        c = sp.candidates_from_table(tabs[0][1], name, country, source="x", confiance="A")
        self.assertEqual(c["pays"].tolist(), ["Poland", "India"])

    def test_csv_and_guess_without_header_hint(self):
        csv = "Raison sociale;Pays\nDupont SAS;France\nMuller GmbH;Allemagne\n"
        tabs = sp.tables_from_bytes(csv.encode(), "f.csv")
        self.assertEqual(sp.guess_columns(tabs[0][1]), ("Raison sociale", "Pays"))

    def test_consolidate_dedupes_and_keeps_best_confidence(self):
        a = sp.candidates_from_text("Acme Ltd;UK\nacme\n12345\n", confiance="D")
        b = sp.candidates_from_table(pd.DataFrame({"n": ["ACME LIMITED", "Zeta SA"], "p": ["", "FR"]}), "n", "p", confiance="A")
        out = sp.consolidate(pd.concat([a, b], ignore_index=True))
        self.assertEqual(len(out), 2)                                    # 12345 écarté, doublons fusionnés
        self.assertEqual(out[out["fournisseur"].str.contains("ACME|Acme", regex=True)].iloc[0]["confiance"], "A")

    def test_assisted_csv_markdown_and_list(self):
        csv = ('```\nfournisseur;pays;type_relation;rang;source_url;citation\n'
               'Acme;Poland;fabricant;1;https://x.org/a;"Acme makes our chairs"\nBeta;India;;;;\n```')
        d = sp.parse_assisted_reply(csv)
        self.assertEqual(d["confiance"].tolist(), ["C", "D"])
        md = "| fournisseur | pays |\n|---|---|\n| Gamma | Italy |\n"
        self.assertEqual(sp.parse_assisted_reply(md)["fournisseur"].tolist(), ["Gamma"])
        self.assertEqual(sp.parse_assisted_reply("- Delta\n- Epsilon")["fournisseur"].tolist(), ["Delta", "Epsilon"])
        self.assertTrue(sp.parse_assisted_reply("AUCUN FOURNISSEUR TROUVÉ").empty)

    def test_prompt_mentions_client(self):
        self.assertIn("IKEA", sp.build_prompt("IKEA"))

    def test_ssrf_blocked(self):
        for u in ("http://127.0.0.1/x", "http://localhost/a", "ftp://example.com/x"):
            with self.assertRaises(sp.SupplyError):
                sp.fetch_url(u)

    def test_output_has_client_first(self):
        sites = pd.DataFrame({"nom_demande": ["Acme"], "name": ["Acme Factory"], "lat": [1.0], "long": [2.0]})
        cands = sp.candidates_from_text("Acme;PL", confiance="A")
        out = sp.build_output("IKEA", sites, cands)
        self.assertEqual(out.columns[0], "client")
        self.assertEqual(out.iloc[0]["client"], "IKEA")
        self.assertEqual(out.iloc[0]["confiance"], "A")
        both = pd.DataFrame({"nom_demande": ["IKEA", "Acme"], "name": ["IKEA Lyon", "Acme Factory"], "lat": [1.0, 2.0], "long": [2.0, 3.0]})
        o2 = sp.build_output("IKEA", both, cands, include_client=True)
        self.assertEqual(list(o2.columns[:2]), ["client", "role"])
        self.assertEqual(o2["role"].tolist(), ["client", "fournisseur"])
        self.assertTrue(pd.isna(o2.iloc[0]["confiance"]))
        self.assertEqual(sp.build_output("IKEA", both, cands)["role"].tolist(), ["fournisseur", "fournisseur"])
        rep = pd.DataFrame({"nom_demande": ["Acme", "Zed"], "statut": ["ok", "aucun résultat"], "n_sites": [1, 0], "message": ["", ""]})
        self.assertEqual(sp.not_found(rep, pd.concat([cands, sp.candidates_from_text("Zed")]), "IKEA")["nom_demande"].tolist(), ["Zed"])


if __name__ == "__main__":
    unittest.main()
