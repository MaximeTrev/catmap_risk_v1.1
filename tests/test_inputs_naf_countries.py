import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from catmap.countries import CountryIndex
from catmap.inputs import InputError, parse_names_text, read_company_list
from catmap.naf import DIVISIONS, SECTIONS, NafClassifier, section_of_division

ROOT = Path(__file__).resolve().parent.parent


class TestInputs(unittest.TestCase):
    def test_pipe_with_header_and_bom(self):
        q, w = read_company_list("﻿Nom de l'entreprise\ntotal\nengie\nIKEA\nApple\n".encode("utf-8"), "a.csv")
        self.assertEqual([x.raw for x in q], ["total", "engie", "IKEA", "Apple"])

    def test_headerless_keeps_first_row(self):
        q, _ = read_company_list(b"engie\nIKEA\n", "a.csv")
        self.assertEqual([x.raw for x in q], ["engie", "IKEA"])      # la v1 aurait perdu « engie »
        q2, _ = read_company_list(b"engie\nIKEA\n", "a.csv", has_header=True)
        self.assertEqual([x.raw for x in q2], ["IKEA"])

    def test_semicolon_wikidata_latin1_dedup(self):
        data = "Société;Wikidata\nTotal;Q154037\nTotal;Q154037\nÉlan;bad\n".encode("cp1252")
        q, w = read_company_list(data, "x.csv")
        self.assertEqual([(x.raw, x.wikidata) for x in q], [("Total", "Q154037"), ("Élan", None)])
        self.assertTrue(any("invalide" in m for m in w))

    def test_real_example_files(self):
        q, _ = read_company_list((ROOT / "examples" / "entreprises_test.csv").read_bytes(), "e.csv")
        self.assertEqual(len(q), 4)
        qx, _ = read_company_list((ROOT / "examples" / "entreprises_135.xlsx").read_bytes(), "e.xlsx")
        self.assertEqual(len(qx), 133)        # 135 lignes, dont 2 doublons (« Apple », « engie ») fusionnés ; aucune ligne perdue à tort

    def test_text_and_errors(self):
        q, _ = parse_names_text("Total;Q154037\n\nIKEA")
        self.assertEqual([(x.raw, x.wikidata) for x in q], [("Total", "Q154037"), ("IKEA", None)])
        with self.assertRaises(InputError):
            read_company_list(b"\n\n", "x.csv")


class TestNaf(unittest.TestCase):
    def test_nomenclature_complete(self):
        self.assertEqual(len(SECTIONS), 21)
        self.assertTrue(all(section_of_division(d) for d in DIVISIONS))
        self.assertEqual((section_of_division("56"), section_of_division("47"), section_of_division("99")), ("I", "G", "U"))

    def test_mapping_integrity(self):
        m = NafClassifier().mapping
        self.assertFalse(m.duplicated(["osm_key", "osm_value"]).any())
        for d in m.naf_division[m.naf_division != ""]:
            self.assertIn(d, DIVISIONS)
        self.assertTrue(set(m.naf_section_override[m.naf_section_override != ""]) <= set(SECTIONS))
        self.assertTrue(set(m.confidence) <= {"high", "medium", "low", "n/a"})

    def test_known_v1_errors_fixed(self):                              # A15
        c = NafClassifier()
        df = pd.DataFrame({"amenity": ["cafe", "pharmacy", None, None], "shop": [None, None, "beverages", "alcohol"]})
        r = c.classify(df)
        self.assertEqual(list(r.naf_section_code), ["I", "G", "G", "G"])

    def test_priority_street_furniture_and_multi_values(self):
        c = NafClassifier()
        r = c.classify(pd.DataFrame({"amenity": ["bench", "restaurant;cafe", None], "shop": ["bakery", None, None],
                                     "industrial": [None, None, "factory"]}))
        self.assertEqual(list(r.naf_source), ["shop=bakery", "amenity=restaurant", "industrial=factory"])
        self.assertEqual(r.naf_section_code[2], "C"); self.assertTrue(pd.isna(r.naf_division_code[2]))

    def test_unmapped_report_and_override(self):
        c = NafClassifier()
        u = c.unmapped(pd.DataFrame({"shop": ["zzz", "zzz", "bakery"], "amenity": [None, "weird", None]}))
        self.assertEqual(set(u.osm_value), {"zzz", "weird"})
        self.assertEqual(c.classify(pd.DataFrame({"shop": ["zzz"]})).naf_section_code[0] is None, True)


class TestCountries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ci = CountryIndex()

    def test_cities(self):
        pts = {"Paris": (2.3522, 48.8566, "France"), "Berlin": (13.405, 52.52, "Allemagne"),
               "Tokyo": (139.69, 35.69, "Japon"), "Sydney": (151.21, -33.87, "Australie")}
        r = self.ci.locate([v[0] for v in pts.values()], [v[1] for v in pts.values()])
        self.assertEqual(list(r.pays), [v[2] for v in pts.values()]); self.assertTrue((r.pays_methode == "within").all())

    def test_enclave_priority(self):
        r = self.ci.locate([12.4534], [41.9029])          # Vatican, enclavé dans l'Italie
        self.assertEqual(r.pays[0], "Vatican")

    def test_snap_and_open_sea_and_invalid(self):
        r = self.ci.locate([-1.43, -30.0, 200.0, np.nan], [46.2, 35.0, 10.0, 10.0], snap_km=50)
        self.assertEqual((r.pays[0], r.pays_methode[0]), ("France", "nearest"))   # île côtière (H9)
        self.assertTrue(r.pays[1:].isna().all())
        r0 = self.ci.locate([-1.43], [46.2], snap_km=0)
        self.assertTrue(pd.isna(r0.pays[0]))

    def test_no_duplicate_rows_and_alignment(self):
        lon = np.array([2.35, 2.35, 13.4]); lat = np.array([48.85, 48.85, 52.5])
        r = self.ci.locate(lon, lat); self.assertEqual(list(r.pays), ["France", "France", "Allemagne"])


if __name__ == "__main__":
    unittest.main()
