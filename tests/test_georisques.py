import unittest

import pandas as pd

from catmap import georisques as gr
from tests.helpers import FakeResponse

RAPPORT = {"adresse": {"libelle": "1 rue X"}, "commune": {"libelle": "Paris"}, "url": "http://pdf",
           "risquesNaturels": {"inondation": {"present": True, "libelleStatutAdresse": "Concerné"},
                               "seisme": {"present": False}},
           "risquesTechnologiques": {"icpe": {"present": True, "libelleStatutCommune": "Présent dans la commune"}}}


class Sess:
    """Réponses par chemin ; mémorise les appels."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, dict(params or {})))
        path = url.replace(gr.BASE_URL, "")
        r = self.routes.get(path)
        return r if r is not None else FakeResponse(404, text="nf")


ROUTES = lambda: {
    "resultats_rapport_risque": FakeResponse(200, RAPPORT),
    "zonage_sismique": FakeResponse(200, {"data": [{"code_zone": "1", "zone_sismicite": "1 - Très faible"}]}),
    "argiles": FakeResponse(200, {"exposition": "Exposition moyenne"}),
    "cavites": FakeResponse(200, {"results": 2, "data": [{"typeCavite": "x", "nom": "Carrière A"}, {"nom": "Cave B"}]}),
    "gaspar/risques": FakeResponse(200, {"results": 1, "data": [{"risques_detail": [{"libelle_risque_long": "Inondation"}, {"libelle_risque_long": "Séisme"}]}]}),
    "radon": FakeResponse(500, text="boom"),
}


class T(unittest.TestCase):
    def test_latlon_order_and_radius(self):
        s = Sess(ROUTES())
        gr.query_dataset(gr.BY_KEY["cavites"], 48.8566, 2.3522, 700, s)
        self.assertEqual(s.calls[0][1], {"latlon": "2.352200,48.856600", "rayon": 700})     # longitude,latitude
        s = Sess(ROUTES())
        gr.query_dataset(gr.BY_KEY["sismique"], 48.8, 2.3, 700, s)
        self.assertNotIn("rayon", s.calls[0][1])

    def test_parsers(self):
        s = Sess(ROUTES())
        q = lambda k: gr.query_dataset(gr.BY_KEY[k], 48.8, 2.3, 1000, s)[0]
        self.assertEqual(q("sismique"), {"gr_sismique": "1 - Très faible"})
        self.assertEqual(q("argiles"), {"gr_argiles": "Exposition moyenne"})       # 1er chemin « rga » en 404 -> repli « argiles »
        v = q("cavites")
        self.assertEqual(v["gr_cavites_nb"], 2)
        self.assertIn("Carrière A", v["gr_cavites_detail"])
        v = q("commune")
        self.assertEqual((v["gr_commune_nb"], v["gr_commune_liste"]), (2, "Inondation | Séisme"))
        v = q("rapport")
        self.assertEqual(v["gr_nat_inondation"], "Oui")
        self.assertEqual(v["gr_nat_seisme"], "Non")
        self.assertEqual(v["gr_nat_inondation_statut"], "Concerné")
        self.assertEqual(v["gr_tech_icpe"], "Oui")
        self.assertEqual(v["gr_commune_rapport"], "Paris")

    def test_report_unexpected_structure_keeps_leaves(self):
        v = gr._flatten_report({"foo": {"bar": 1, "baz": "x"}})
        self.assertEqual(v["gr_rapport_foo_bar"], 1)

    def test_errors_are_statuses_not_exceptions(self):
        s = Sess(ROUTES())
        v, err, _ = gr.query_dataset(gr.BY_KEY["mvt"], 48.8, 2.3, 1000, s)
        self.assertEqual(err, "chemin introuvable (HTTP 404)")
        self.assertEqual(v, {})

    def test_enrich_dedup_invalid_and_status(self):
        s = Sess(ROUTES())
        df = pd.DataFrame({"name": ["a", "b", "c", "d"], "lat": [48.8566, 48.8566, "x", 45.0], "long": [2.3522, 2.3522, 2.0, 999]})
        out, summ = gr.enrich_points(df, "lat", "long", ["sismique", "cavites", "mvt"], 1000, 10, s, workers=2)
        self.assertEqual(summ["interroges"], 1)                       # doublon fusionné, invalides ignorés
        self.assertEqual(summ["invalides"], 2)
        self.assertEqual(out.loc[0, "gr_sismique"], "1 - Très faible")
        self.assertEqual(out.loc[0, "gr_cavites_nb"], 2)
        self.assertEqual(out.loc[0, "gr_statut"], "partiel")           # mvt en 404
        self.assertIn("mvt", out.loc[0, "gr_erreurs"])
        self.assertEqual(out.loc[2, "gr_statut"], "coordonnées invalides")
        self.assertEqual(len(out), 4)
        self.assertEqual(len([c for c in s.calls if c[0].endswith("zonage_sismique")]), 1)

    def test_max_points(self):
        s = Sess(ROUTES())
        df = pd.DataFrame({"lat": [48.0 + i / 100 for i in range(5)], "long": [2.0] * 5})
        out, summ = gr.enrich_points(df, "lat", "long", ["sismique"], 1000, 2, s, workers=1)
        self.assertEqual((summ["interroges"], summ["non_interroges"]), (2, 3))
        self.assertEqual(list(out["gr_statut"]).count("non interrogé"), 3)

    def test_probe(self):
        rows = gr.probe(48.8, 2.3, ["sismique", "mvt"], 1000, Sess(ROUTES()))
        self.assertEqual(rows[0]["statut"], "ok")
        self.assertIn("404", rows[1]["statut"])


if __name__ == "__main__":
    unittest.main()
