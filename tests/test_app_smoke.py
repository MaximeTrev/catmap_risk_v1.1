"""Test de fumée de app.py avec un faux module Streamlit (exécuté seulement si Streamlit n'est pas installé).

Il exécute réellement les 3 pages, une recherche (Overpass simulé), les téléchargements et le
croisement vigilance, pour détecter fautes de frappe, colonnes manquantes et exceptions.
Avec Streamlit installé, utiliser plutôt ``streamlit.testing.v1.AppTest`` (voir README).
"""
import importlib
import importlib.util
import sys
import types
import unittest
from unittest import mock

from tests.helpers import node, payload, square, way

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


class Container:
    """Comme st.columns / st.tabs / st.sidebar : les méthodes déléguent à ``st`` ; utilisable avec ``with``."""
    def __init__(self, st): self._st = st
    def __getattr__(self, name): return getattr(self._st, name)
    def __enter__(self): return self
    def __exit__(self, *a): return False


def make_fakes(script):
    st = types.ModuleType("streamlit")
    st.session_state = SS()
    st.secrets = mock.MagicMock(); st.secrets.items.side_effect = RuntimeError("no secrets")
    st.calls = []
    M = mock.MagicMock

    def rec(name, ret=None):
        def f(*a, **k):
            st.calls.append((name, a, k)); return ret
        return f

    def deco(*a, **k):
        def wrap(fn):
            fn.clear = lambda: None
            return fn
        return wrap(a[0]) if a and callable(a[0]) else wrap

    st.cache_resource, st.cache_data = deco, deco
    st.radio = lambda label, options=None, **k: script.get(k.get("key") or label, options[0])
    st.text_input = lambda label, **k: script.get(k.get("key"), "")
    st.text_area = lambda label, **k: script.get(k.get("key"), "")
    st.file_uploader = lambda *a, **k: script.get(k.get("key"))
    st.checkbox = lambda label, value=False, **k: value
    st.multiselect = lambda label, options=None, default=None, **k: script.get(k.get("key"), default or [])
    st.slider = lambda label, mn, mx, value, *a, **k: value
    st.selectbox = lambda label, options, **k: options[0]
    st.button = lambda label, **k: script.get("button:" + label, False) and not k.get("disabled")
    st.download_button = lambda *a, **k: (st.calls.append(("download", a[0], a[1].__len__() if hasattr(a[1], "__len__") else 0)), False)[1]
    st.columns = lambda n, **k: [Container(st) for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda labels: [Container(st) for _ in labels]
    st.empty = lambda: M()
    st.expander = lambda *a, **k: Container(st)
    st.sidebar = Container(st)
    for n in ("title", "subheader", "caption", "markdown", "info", "warning", "error", "success", "divider",
              "dataframe", "plotly_chart", "set_page_config", "image", "metric"):
        setattr(st, n, rec(n))
    st.column_config = M()
    plotly = types.ModuleType("plotly"); px = types.ModuleType("plotly.express"); px.pie = M(); plotly.express = px
    folium = M(); sf = types.ModuleType("streamlit_folium"); sf.st_folium = rec("st_folium")
    comp = types.ModuleType("streamlit.components"); comp_v1 = types.ModuleType("streamlit.components.v1")
    comp_v1.html = rec("components_html"); comp.v1 = comp_v1; st.components = comp
    st.spinner = lambda *a, **k: Container(st)
    st.container = lambda *a, **k: Container(st)
    return {"streamlit.components": comp, "streamlit.components.v1": comp_v1, "streamlit": st, "plotly": plotly, "plotly.express": px, "folium": folium, "folium.plugins": folium.plugins,
            "streamlit_folium": sf}, st


@unittest.skipIf(HAS_STREAMLIT, "Streamlit installé : utiliser AppTest")
class TestAppSmoke(unittest.TestCase):
    def load(self, script):
        mods, st = make_fakes(script)
        # on n'installe/retire QUE les faux modules (patch.dict purgerait aussi numpy/pandas)
        for k, v in mods.items():
            sys.modules[k] = v
            self.addCleanup(sys.modules.pop, k, None)
        sys.modules.pop("app", None)
        app = importlib.import_module("app")
        self.addCleanup(lambda: sys.modules.pop("app", None))
        return app, st

    def fake_cyclones(self):
        from catmap import cyclones as cy
        from tests.test_cyclones import FORECAST, NOW, OBSERVED
        storms = cy.build_storms(cy.parse_forecast(FORECAST), cy.parse_observed(OBSERVED), now=NOW)
        stack = mock.patch.multiple(cy, fetch_storms=mock.Mock(return_value=storms),
                                    fetch_wind_radii=mock.Mock(return_value={}),
                                    attach_precipitation=mock.Mock(return_value=True))
        return stack

    def fake_fetch(self):
        from catmap.overpass import FetchResult
        els = [way(1, "IKEA Lyon", square(4.85, 45.75, 0.002), shop="furniture"),
               node(2, "IKEA Berlin", 13.4, 52.5, shop="furniture"), node(3, "Café IKEA", 2.35, 48.85, amenity="cafe")]
        return mock.patch("catmap.overpass.OverpassClient.fetch",
                          return_value=FetchResult(payload(els), "http://fake", "2026-09-30T20:00:00Z", False, 1, 0.1))

    def test_geolocalisation_flow_and_other_pages(self):
        script = {"Mode de saisie": "Un nom d'entreprise", "in_name": "IKEA SA", "button:Lancer la recherche": True}
        app, st = self.load(script)
        with self.fake_fetch(), mock.patch.dict("os.environ", {"CATMAP_CACHE": "0"}):
            app.main()
        res = st.session_state["result"]
        self.assertEqual(len(res.sites), 3)
        names = [c[0] for c in st.calls]
        self.assertIn("dataframe", names); self.assertIn("st_folium", names); self.assertIn("download", names)
        self.assertFalse([c for c in st.calls if c[0] == "error"])

        # Page vigilance croisée avec les sites (clé API factice, réseau simulé)
        from catmap import vigilance as vg
        from tests.test_vigilance import GEO, PAYLOAD
        script["Navigation"] = "Cartographies avancées"
        st.calls.clear()
        with mock.patch.dict("os.environ", {"METEOFRANCE_API_KEY": "K"}), \
                mock.patch.object(vg, "fetch_vigilance", return_value=PAYLOAD), \
                mock.patch.object(vg, "load_departements", return_value=vg.DepartementIndex(GEO)):
            app.main()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertTrue([c for c in st.calls if c[0] == "metric"])

        script["Navigation"] = "Temps réel"
        st.calls.clear()
        with self.fake_cyclones(), mock.patch.dict("os.environ", {"METEOFRANCE_API_KEY": "K"}), \
                mock.patch.object(vg, "fetch_vigilance", return_value=PAYLOAD), \
                mock.patch.object(vg, "load_departements", return_value=vg.DepartementIndex(GEO)):
            app.main()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)

    def test_cyclone_section_renders_map_and_table(self):
        app, st = self.load({"Navigation": "Temps réel"})
        with self.fake_cyclones(), mock.patch.dict("os.environ", {}, clear=False):
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        html_calls = [c for c in st.calls if c[0] == "components_html"]
        self.assertEqual(len(html_calls), 1)
        page = html_calls[0][1][0]
        self.assertIn("Melissa", page)
        self.assertIn("Maximum Sustained Winds", page)
        self.assertIn("L.control.layers", page)
        self.assertTrue([c for c in st.calls if c[0] == "dataframe"])
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)

    def test_cyclone_risk_upload_and_analysis(self):
        from catmap import cyclone_risk as cr

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long\nA;16,0;-75,0\nB;48,85;2,35\n".encode()

        app, st = self.load({"Navigation": "Temps réel", "cy_csv": Up(), "button:Analyser le risque": True})
        with self.fake_cyclones(), mock.patch.object(cr, "site_precipitation", side_effect=lambda pts, **k: [7.0] * len(pts)):
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        res = st.session_state["cy_risk"]
        self.assertEqual(list(res["impact"]), [12, 0])
        self.assertEqual(res.iloc[0]["total_precipitation"], 7.0)
        self.assertTrue([c for c in st.calls if c[0] == "download"])
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        page = [c for c in st.calls if c[0] == "components_html"][-1][1][0]
        self.assertIn('"points_analysed": true', page)       # les points du CSV sont sur la carte, colorés par impact
        self.assertIn("impact 12 h", page)

    def test_points_shown_on_map_before_analysis(self):
        class Up:
            name = "pts.csv"
            def getvalue(self): return b"nom;lat;long\nA;16,0;-75,0\n"
        app, st = self.load({"Navigation": "Temps réel", "cy_csv": Up()})
        with self.fake_cyclones():
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        page = [c for c in st.calls if c[0] == "components_html"][-1][1][0]
        self.assertIn('"points_analysed": false', page)
        self.assertIn('"tip": "A"', page)

    def test_cyclone_risk_button_hidden_without_upload(self):
        app, st = self.load({"Navigation": "Temps réel", "button:Analyser le risque": True})
        with self.fake_cyclones():
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        self.assertNotIn("cy_risk", st.session_state)

    def test_vigilance_page_with_uploaded_file_without_geolocation(self):
        from catmap import vigilance as vg
        from tests.test_vigilance import GEO, PAYLOAD

        class Up:
            name = "export.csv"
            # export « périmètre » : pas de pays_iso2, virgule décimale, colonnes lat / long
            def getvalue(self):
                return "name;lat;long\nBoulangerie;45,5;1,5\nHors France;10;10\nSans coord;;\n".encode()

        app, st = self.load({"Navigation": "Cartographies avancées", "vg_upload": Up()})
        with mock.patch.dict("os.environ", {"METEOFRANCE_API_KEY": "K"}), \
                mock.patch.object(vg, "fetch_vigilance", return_value=PAYLOAD), \
                mock.patch.object(vg, "load_departements", return_value=vg.DepartementIndex(GEO)):
            app.main()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertTrue([c for c in st.calls if c[0] == "warning"])        # ligne sans coordonnées signalée
        metrics = {c[1][0]: c[1][1] for c in st.calls if c[0] == "metric"}
        self.assertEqual(metrics["Hors France (non traités)"], 2 - metrics["Sites en France"])
        self.assertFalse([c for c in st.calls if c[0] == "info"])           # plus de « lancez d'abord une géolocalisation »

    def test_vigilance_page_without_source_asks_for_file(self):
        app, st = self.load({"Navigation": "Cartographies avancées"})
        app.main()
        infos = [c[1][0] for c in st.calls if c[0] == "info"]
        self.assertTrue(any("Chargez un fichier" in i for i in infos))

    def test_cyclone_service_down_shows_error_not_crash(self):
        from catmap import cyclones as cy
        app, st = self.load({"Navigation": "Temps réel"})
        with mock.patch.object(cy, "fetch_storms", side_effect=cy.CycloneError("hors ligne")):
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        errs = [c for c in st.calls if c[0] == "error"]
        self.assertEqual(len(errs), 1)
        self.assertIn("hors ligne", errs[0][1][0])
        self.assertTrue([c for c in st.calls if c[0] == "info"])      # l'onglet vigilance reste utilisable

    def test_empty_search_does_not_crash(self):
        from catmap.overpass import FetchResult
        app, st = self.load({"Mode de saisie": "Un nom d'entreprise", "in_name": "Rien", "button:Lancer la recherche": True})
        with mock.patch("catmap.overpass.OverpassClient.fetch", return_value=FetchResult(payload([]), "x", "", False, 1, 0.0)):
            app.main()
        self.assertEqual(len(st.session_state["result"].sites), 0)
        self.assertTrue([c for c in st.calls if c[0] == "info"])
        self.assertFalse([c for c in st.calls if c[0] == "error"])

    def test_no_credentials_page_is_informative(self):
        app, st = self.load({"Navigation": "Temps réel"})
        with self.fake_cyclones(), mock.patch.dict("os.environ", {}, clear=False):
            for k in ("METEOFRANCE_API_KEY", "METEOFRANCE_CLIENT_ID", "METEOFRANCE_CLIENT_SECRET"):
                import os; os.environ.pop(k, None)
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        self.assertTrue([c for c in st.calls if c[0] == "info"])


if __name__ == "__main__":
    unittest.main()
