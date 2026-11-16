import io
import unittest

import numpy as np
import pandas as pd

from catmap import damage as dm


def hazus_csv() -> bytes:
    cols = ["", "Occupancy", "DmgFnId", "Source", "Description", "ft04m", "ft03m", "ft02m", "ft01m", "ft00", "ft01", "ft02", "ft03", "ft04",
            "ft05", "ft06", "Comment", "ft00_5", "ft01_5", "Source_Table", "Occupy_Class", "Cover_Class"]
    rows = [
        [1, "COM1", 10, "USACE", "retail", 0, 0, 0, 0, 10, 20, 30, 40, 50, 60, 70, "", 5, 15, "flBldgStructDmgFn", "COM", "Bldg"],
        [2, "COM1", 11, "FIA", "retail 2", 0, 0, 0, 0, 20, 30, 40, 50, 60, 70, 80, "", 10, 25, "flBldgStructDmgFn", "COM", "Bldg"],
        [3, "COM1", 12, "X", "contenu", 0, 0, 0, 0, 90, 90, 90, 90, 90, 90, 90, "", None, None, "flBldgContDmgFn", "COM", "Cont"],
        [4, "IND1", 20, "USACE", "heavy", 0, 0, 0, 0, 0, 10, 20, 30, 40, 50, 60, "", None, None, "flBldgStructDmgFn", "IND", "Bldg"],
    ]
    return pd.DataFrame(rows, columns=cols).to_csv(index=False).encode("utf-8-sig")


class TestHazus(unittest.TestCase):
    def test_parse_and_curve(self):
        hz = dm.parse_hazus(hazus_csv())
        self.assertEqual(set(hz["occupancies"]), {"COM1", "IND1"})          # le contenu (Cont) est exclu
        c = dm.hazus_curve(hz, "COM1")
        # médiane de 2 courbes à 0 ft : (10 + 20) / 2 = 15 % ; pieds → mètres
        self.assertAlmostEqual(float(c.at(0.0)), 0.15, places=6)
        self.assertAlmostEqual(float(c.at(1 * dm.FT)), 0.25, places=6)
        self.assertAlmostEqual(float(c.at(0.5 * dm.FT)), 0.075, places=6)   # 5 et 10 % à 0,5 ft → médiane 7,5 %
        self.assertEqual(c.n_curves, 2)

    def test_floor_height_shifts_curve(self):
        hz = dm.parse_hazus(hazus_csv())
        c = dm.hazus_curve(hz, "IND1", floor_height_m=0.5)
        self.assertAlmostEqual(float(c.at(0.5)), 0.0, places=6)                 # eau au niveau du plancher
        self.assertAlmostEqual(float(c.at(0.5 + dm.FT)), 0.10, places=6)

    def test_riverine_defaults_filter(self):
        hz = dm.parse_hazus(hazus_csv(), riverine_ids_bytes=b"DmgFnId,Source\n11,FIA\n")
        self.assertEqual(hz["ids"], {11})
        c = dm.hazus_curve(hz, "COM1")
        self.assertAlmostEqual(float(c.at(0.0)), 0.20, places=6)
        self.assertEqual(c.n_curves, 1)

    def test_bad_file(self):
        with self.assertRaises(dm.DamageError):
            dm.parse_hazus(b"a,b\n1,2\n")
        with self.assertRaises(dm.DamageError):
            dm.hazus_curve(dm.parse_hazus(hazus_csv()), "XXX")

    def test_depth_column_names(self):
        self.assertEqual(dm._hazus_depth_ft("ft04m"), -4.0)
        self.assertEqual(dm._hazus_depth_ft("ft24"), 24.0)
        self.assertEqual(dm._hazus_depth_ft("ft01_5"), 1.5)
        self.assertIsNone(dm._hazus_depth_ft("Comment"))


class TestJrc(unittest.TestCase):
    def test_long_csv(self):
        csv = "continent,classe,hauteur_m,taux\nEurope,industrial,0,0\nEurope,industrial,1,0.3\nEurope,industrial,2,0.6\n"
        j = dm.parse_jrc(csv.encode(), "jrc.csv")
        c = dm.jrc_curve(j, "Europe", "industrial")
        self.assertAlmostEqual(float(c.at(1.5)), 0.45, places=6)
        self.assertAlmostEqual(float(c.at(10)), 0.6, places=6)                  # plafond
        with self.assertRaises(dm.DamageError):
            dm.jrc_curve(j, "Asia", "industrial")

    def test_max_values_csv(self):
        j = dm.parse_jrc(b"pays,classe,eur_m2\nFrance,industrial,450\n", "max.csv")
        self.assertEqual(dm.jrc_max_value(j, "france", "industrial"), 450.0)

    def test_excel_blocks(self):
        grid1 = pd.DataFrame([["Europe", None, None, None], ["Flood depth [m]", "Residential buildings", "Commercial buildings", "Industrial buildings"],
                              [0, 0, 0, 0], [1, 0.4, 0.3, 0.2], [2, 0.7, 0.5, 0.4], [None, None, None, None],
                              ["Asia", None, None, None], ["Flood depth [m]", "Residential buildings", "Commercial buildings", "Industrial buildings"],
                              [0, 0, 0, 0], [1, 0.1, 0.1, 0.1], [2, 0.2, 0.2, 0.2]])
        grid2 = pd.DataFrame([["Country", "Residential", "Commercial", "Industrial"], ["France", 800, 600, 450], ["Italy", 700, 500, 400]])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf) as xw:
            grid1.to_excel(xw, sheet_name="Damage functions", header=False, index=False)
            grid2.to_excel(xw, sheet_name="Max damage", header=False, index=False)
        j = dm.parse_jrc(buf.getvalue(), "jrc.xlsx")
        self.assertAlmostEqual(float(dm.jrc_curve(j, "Europe", "industrial").at(1.0)), 0.2)
        self.assertAlmostEqual(float(dm.jrc_curve(j, "Asia", "commercial").at(2.0)), 0.2)
        self.assertEqual(dm.jrc_max_value(j, "France", "commercial"), 600.0)

    def test_official_layout(self):
        """Mise en page du classeur officiel JRC : continents en colonnes, catégories en blocs de lignes, pays en lignes."""
        hdr = [["Damage\nclass", "Flood depth,\n[m]", "Damage function"] + [None] * 6 + ["Standard deviation"] + [None] * 6,
               [None, None, "EUROPE", "North AMERICA", "Centr&South\nAMERICA", "ASIA", "AFRICA", "OCEANIA", "GLOBAL",
                "EUROPE", "North AMERICA", "Centr&South\nAMERICA", "ASIA", "AFRICA", "OCEANIA", "GLOBAL"]]
        body = [["Industrial buildings", 0, 0, 0.0257, 0, 0, 0, "-", 0] + [0] * 7, [None, 1, 0.27, 0.51, 0.89, 0.48, 0.25, "-", 0.48] + [0.1] * 7,
                [None, 2, 0.52, 0.74, 1, 0.72, 0.49, "-", 0.69] + [0.1] * 7,
                ["Transport", 0, 0, "-", 0, 0, "-", "-", 0] + [0] * 7, [None, 1, 0.54, "-", 0.17, 0.57, "-", "-", 0.43] + [0.1] * 7]
        grid = pd.DataFrame([[None] * 16] + hdr + body)
        mx = pd.DataFrame([[None, "Building based", None, None, "Land-use based", "Object based"],
                           ["Country", "Max Damage Structure", "Max Damage Content", "Total", "Total", "Total"],
                           [None, "(EUR/m2, 2010)", "(EUR/m2, 2010)", "(EUR/m2, 2010)", "(EUR/m2, 2010)", "(EUR/object, 2010)"],
                           ["France", 350.0, 520.0, 870.0, 260.0, 1.0], ["Vietnam", 115.0, 172.0, 287.0, 86.0, 1.0]])
        data = pd.DataFrame([[None, None, None, "RES"], ["Country", "ISO A3 code", "GDP", "RES"], ["France", "FRA", 40000, 1], ["Vietnam", "VNM", 1000, 1]])
        isot = pd.DataFrame([["Country", "A 2", "A 3", "Number"], ["FRANCE", "FR", "FRA", 250], ["VIET NAM", "VN", "VNM", 704]])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf) as xw:
            grid.to_excel(xw, sheet_name="Damage functions", header=False, index=False)
            mx.to_excel(xw, sheet_name="MaxDamage-Industrial", header=False, index=False)
            data.to_excel(xw, sheet_name="MaxDamage-Data", header=False, index=False)
            isot.to_excel(xw, sheet_name="ISO_Table", header=False, index=False)
        j = dm.parse_jrc(buf.getvalue(), "JRC_global.xlsx")
        self.assertAlmostEqual(float(dm.jrc_curve(j, "Europe", "industrial").at(1.0)), 0.27)
        self.assertAlmostEqual(float(dm.jrc_curve(j, "South America", "industrial").at(1.0)), 0.89)
        self.assertAlmostEqual(float(dm.jrc_curve(j, "Asia", "transport").at(1.0)), 0.57)
        with self.assertRaises(dm.DamageError):                                 # « - » dans le fichier : pas de courbe
            dm.jrc_curve(j, "Oceania", "industrial")
        self.assertEqual(dm.jrc_max_value(j, "France", "industrial"), 870.0)
        self.assertEqual(dm.jrc_max_value(j, "FR", "industrial", "struct"), 350.0)
        self.assertEqual(dm.jrc_max_value(j, "VNM", "industrial"), 287.0)
        self.assertEqual(dm.jrc_max_value(j, "VN", "industrial"), 287.0)

    def test_percent_scale_detected(self):
        csv = "continent,classe,hauteur_m,taux\nEurope,commercial,0,0\nEurope,commercial,1,50\n"
        j = dm.parse_jrc(csv.encode(), "jrc.csv")
        self.assertAlmostEqual(float(dm.jrc_curve(j, "Europe", "commercial").at(1.0)), 0.5)


class TestEstimate(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"name": ["a", "b", "c"], "type": ["Usine", "Magasin", "Usine"],
                                "debordement_RP10_raw": [0.0, 1.0, np.nan], "debordement_RP100_raw": [1.0, 2.0, np.nan],
                                "valeur": [1000.0, 2000.0, 500.0]})
        self.curve = dm.Curve([0, 1, 2], [0, 0.2, 0.6], "test")

    def test_losses_and_aal(self):
        cls = dm.asset_class_series(self.df, "type", "commercial")
        self.assertEqual(list(cls), ["industrial", "commercial", "industrial"])
        vals = dm.exposed_value(self.df, "col", value_col="valeur")
        out, s = dm.estimate_damages(self.df, {10: "debordement_RP10_raw", 100: "debordement_RP100_raw"},
                                     {"industrial": self.curve, "commercial": self.curve}, cls, vals, "test")
        self.assertEqual(list(out["dommage_RP100_ratio"].round(3))[:2], [0.2, 0.6])
        self.assertAlmostEqual(out["dommage_RP100_eur"].iloc[0], 200.0)
        self.assertAlmostEqual(out["dommage_RP10_eur"].iloc[0], 0.0)           # hauteur 0 → aucun dommage
        self.assertTrue(np.isnan(out["dommage_RP100_eur"].iloc[2]))             # hauteur inconnue → pas de perte inventée
        # PAM point a : trapèze entre p=0,01 (200 €) et p=0,1 (0 €) = 0,09 × 100 = 9
        self.assertAlmostEqual(out["dommage_PAM_eur"].iloc[0], 9.0, places=6)
        self.assertEqual(s["pam_bornes"], (10, 100))
        self.assertEqual(s["sans_hauteur"], 1)

    def test_value_modes(self):
        df = pd.DataFrame({"s": ["100", "2 000"], "pays": ["France", "Italy"]})
        self.assertEqual(list(dm.exposed_value(df, "area", area_col="s", unit_value=10)[:1]), [1000.0])
        self.assertEqual(list(dm.exposed_value(df, "const", const=5.0)), [5.0, 5.0])
        j = {"maxdmg": {("france", "industrial"): 450.0}, "curves": {}}
        cl = pd.Series(["industrial", "industrial"])
        v = dm.exposed_value(df, "jrc_area", area_col="s", country_col="pays", classes=cl, jrc=j)
        self.assertEqual(v.iloc[0], 45000.0)
        self.assertTrue(np.isnan(v.iloc[1]))                                    # pays absent : valeur vide, pas inventée

    def test_missing_curve_reported(self):
        cls = pd.Series(["industrial", "transport", "industrial"])
        _, s = dm.estimate_damages(self.df, {10: "debordement_RP10_raw"}, {"industrial": self.curve}, cls,
                                   pd.Series([1.0, 1.0, 1.0]), "t")
        self.assertEqual(s["classes_sans_courbe"], ["transport"])

    def test_classes_for_map(self):
        e = dm.class_edges(pd.Series([0, 10, 20, 30, 40]), "eur")
        self.assertEqual(len(e), 3)
        self.assertEqual(dm.damage_class(0, e), 0)
        self.assertEqual(dm.damage_class(1e9, e), 4)
        self.assertIsNone(dm.damage_class(float("nan"), e))


if __name__ == "__main__":
    unittest.main()


class TestLocalFiles(unittest.TestCase):
    def test_find_local_case_insensitive_and_alias(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "HAZ_riverine_ids.csv").write_text("DmgFnId\n1\n")
            (Path(d) / "JRC_global_flood_depth-damage_functions__30102017.xlsx").write_bytes(b"x")
            with mock.patch.object(dm, "DATA_DIRS", (Path(d),)), mock.patch.object(dm, "CACHE_DIR", Path(d) / "none"):
                self.assertEqual(dm.find_local("riverine_ids.csv").name, "HAZ_riverine_ids.csv")
                self.assertIsNone(dm.find_local("haz_fl_dept.csv"))
                with self.assertRaises(dm.DamageError):                    # fichier illisible : erreur claire, pas de plantage muet
                    dm.load_jrc_local()


class TestContinents(unittest.TestCase):
    def test_continent_from_country(self):
        jrc = {"iso": {"fra": "france", "fr": "france", "vnm": "vietnam", "vn": "vietnam", "viet nam": "vietnam"}}
        df = pd.DataFrame({"c": ["FR", "fra", "France", "Viet Nam", "VN", "US", "ZZ", None]})
        s, unknown = dm.continent_series(df, "c", jrc, "Europe")
        self.assertEqual(list(s), ["Europe", "Europe", "Europe", "Asia", "Asia", "North America", "Europe", "Europe"])
        self.assertEqual(unknown, 2)                                          # ZZ et vide → défaut
        s0, u0 = dm.continent_series(df, None, jrc, "Asia")
        self.assertEqual(set(s0), {"Asia"})
        self.assertEqual(u0, 0)

    def test_global_fallback_and_per_point_curves(self):
        jrc = {"curves": {("Europe", "industrial"): ([0, 2], [0, 0.5]), ("Global", "industrial"): ([0, 2], [0, 0.4]),
                          ("Global", "commercial"): ([0, 2], [0, 0.3])}, "maxdmg": {}}
        c = dm.jrc_curve_for(jrc, "Oceania", "industrial")
        self.assertEqual(c.continent, "Global")
        self.assertIn("repli", c.label)
        self.assertEqual(dm.jrc_curve_for(jrc, "Europe", "industrial").continent, "Europe")
        with self.assertRaises(dm.DamageError):
            dm.jrc_curve_for(jrc, "Europe", "residential")
        df = pd.DataFrame({"debordement_RP100_raw": [2.0, 2.0]})
        cont = pd.Series(["Europe", "Oceania"])
        cls = pd.Series(["industrial", "industrial"])
        curves = {"Europe|industrial": dm.jrc_curve_for(jrc, "Europe", "industrial"), "Oceania|industrial": dm.jrc_curve_for(jrc, "Oceania", "industrial")}
        out, s = dm.estimate_damages(df, {100: "debordement_RP100_raw"}, curves, cls, pd.Series([100.0, 100.0]), "JRC", cont)
        self.assertEqual(list(out["dommage_RP100_eur"]), [50.0, 40.0])
        self.assertEqual(s["repli_global"], 1)
