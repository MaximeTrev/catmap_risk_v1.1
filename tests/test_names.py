import re
import unittest
from catmap.names import (NameMatcher, clean_company_name, normalize_name, server_regex, valid_wikidata)


class TestClean(unittest.TestCase):
    def test_legal_forms_removed(self):
        for raw, exp in [("Bouygues SA", "Bouygues"), ("Carrefour SCA", "Carrefour"), ("L'Oréal SA", "L'Oréal"),
                         ("Airbus SE", "Airbus"), ("Apple Inc.", "Apple"), ("BMW AG", "BMW"),
                         ("Foo GmbH & Co. KG", "Foo"), ("Air Liquide S.A.", "Air Liquide"),
                         ("Alpha SA de CV", "Alpha"), ("Tesco PLC", "Tesco")]:
            self.assertEqual(clean_company_name(raw), exp, raw)

    def test_no_substring_damage(self):
        # A3 : la v1 produisait « Groupenofi » et « Totalnte »
        for raw in ["Groupe Sanofi", "Total Sante", "Sephora France", "Eiffage", "IKEA", "Kiss se"]:
            self.assertEqual(clean_company_name(raw), raw)

    def test_never_empty(self):
        self.assertEqual(clean_company_name("SA"), "SA")


class TestRegex(unittest.TestCase):
    def srv(self, name, cand):
        return re.search(server_regex(name), cand, re.I) is not None

    def test_server_regex_tolerant(self):
        self.assertTrue(self.srv("Leroy Merlin", "Leroy-Merlin Paris"))
        self.assertTrue(self.srv("Leroy Merlin", "LeroyMerlin"))
        self.assertTrue(self.srv("L'Oréal", "L’Oreal Paris"))
        self.assertTrue(self.srv("Oreal", "L’Oréal"))
        self.assertTrue(self.srv("Skoda", "Škoda Auto"))
        self.assertFalse(self.srv("IKEA", "Carrefour"))

    def test_server_regex_is_safe(self):
        rx = server_regex('Evil"]; out; //\\')
        self.assertNotRegex(rx, r'["\\;\]]')

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError):
            server_regex("  !!  ")


class TestMatcher(unittest.TestCase):
    def test_whole_word_scoring(self):
        m = NameMatcher("IKEA")
        sc = lambda c: (m.match_tags({"name": c}) or None) and m.match_tags({"name": c}).score
        self.assertEqual(sc("IKEA"), 0.9)
        self.assertEqual(sc("Ikea Lyon"), 0.75)
        self.assertEqual(sc("Magasin IKEA"), 0.6)
        self.assertEqual(sc("Ikea-Bordeaux"), 0.75)
        self.assertIsNone(m.match_tags({"name": "xxIKEAxxx"}))
        self.assertIsNone(m.match_tags({"name": "Pineapple"}))

    def test_operator_and_wikidata(self):
        self.assertIsNone(NameMatcher("Total").match_tags({"operator": "Total"}))
        self.assertEqual(NameMatcher("Total").match_tags({"operator": "Total"}, include_operator=True).score, 0.7)
        mi = NameMatcher("Total", "Q154037").match_tags({"brand:wikidata": "q154037", "name": "Station"})
        self.assertEqual((mi.field, mi.score), ("brand:wikidata", 1.0))

    def test_compact_equivalence(self):
        self.assertTrue(NameMatcher("LeroyMerlin").match_tags({"name": "Leroy Merlin"}).exact)
        self.assertTrue(NameMatcher("H&M").match_tags({"brand": "H & M"}).exact)

    def test_wikidata_validation(self):
        self.assertEqual(valid_wikidata("q123"), "Q123")
        self.assertIsNone(valid_wikidata('Q1"]'))
        self.assertIsNone(valid_wikidata("Q1;x"))


if __name__ == "__main__":
    unittest.main()
