from pathlib import Path
import tempfile
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
    st.date_input = lambda label, value=None, **k: value
    st.multiselect = lambda label, options=None, default=None, **k: script.get(k.get("key"), default or [])
    st.slider = lambda label, mn, mx, value, *a, **k: value
    if not hasattr(st, "number_input"):
        st.number_input = lambda label, *a, **k: k.get("value", 0.0)
    st.line_chart = rec("line_chart")
    st.altair_chart = rec("altair_chart")
    st.rerun = rec("rerun")
    st.progress = lambda *a, **k: mock.MagicMock()
    st.selectbox = lambda label, options, **k: script.get(k.get("key"), options[0])
    st.button = lambda label, **k: script.get("button:" + label, False) and not k.get("disabled")
    st.download_button = lambda *a, **k: (st.calls.append(("download", a[0], a[1].__len__() if hasattr(a[1], "__len__") else 0)), False)[1]
    st.columns = lambda n, **k: [Container(st) for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda labels: (st.calls.append(("tabs", list(labels))), [Container(st) for _ in labels])[1]
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
    comp_v1.declare_component = lambda *a, **k: (lambda **kw: None)
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
        from catmap import meteoalarm as ma
        import contextlib
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.multiple(cy, fetch_storms=mock.Mock(return_value=storms),
                                                fetch_wind_radii=mock.Mock(return_value={}),
                                                attach_precipitation=mock.Mock(return_value=True)))
        # aucun accès réseau pour l'onglet Vigilance Europe
        import pandas as pd
        stack.enter_context(mock.patch.object(ma, "fetch_all", return_value=(pd.DataFrame(columns=ma.COLUMNS), {}, {})))
        stack.enter_context(mock.patch.object(ma, "load_geocodes", side_effect=ma.MeteoAlarmError("hors ligne")))
        # aucun accès réseau pour « Suivi évènements climatiques »
        from catmap import events as evt
        from tests.test_events import ACT, CAPS
        stack.enter_context(mock.patch.object(evt, "fetch_activations",
                                              return_value=(evt.parse_activations(ACT["results"]), [])))
        stack.enter_context(mock.patch.object(evt, "discover_wms",
                                              return_value=("http://wms.test", evt.parse_capabilities(CAPS), [])))
        stack.enter_context(mock.patch.object(evt, "burnt_areas", return_value=(None, [])))
        stack.enter_context(mock.patch.object(evt, "fetch_layer_details", return_value=[]))
        stack.enter_context(mock.patch.object(evt, "fetch_activation_detail", return_value=(None, [])))
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
        script["Navigation"] = "Temps réel"
        st.calls.clear()
        with self.fake_cyclones(), mock.patch.dict("os.environ", {"METEOFRANCE_API_KEY": "K"}), \
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

    def test_events_section_renders_without_error(self):
        app, st = self.load({"Navigation": "Temps réel"})
        with self.fake_cyclones():
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertTrue([c for c in st.calls if c[0] == "download"])          # CSV des activations
        self.assertTrue([c for c in st.calls if c[0] == "st_folium"])

    def test_activation_contours_and_legend(self):
        from catmap import events as evt
        from tests.test_events import DETAIL, SLD, _gj
        app, st = self.load({"Navigation": "Temps réel"})
        with self.fake_cyclones(), \
                mock.patch.object(evt, "fetch_activation_detail", return_value=(DETAIL, [])), \
                mock.patch.object(evt, "fetch_geojson", return_value=(_gj(), "")), \
                mock.patch.object(evt, "fetch_text", return_value=SLD):
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        md = " ".join(str(c[1]) for c in st.calls if c[0] == "markdown")
        self.assertIn("Flooded area", md)                       # légende issue du SLD
        self.assertIn("#0000ff", md)

    def test_activation_contours_cross_follows_selected_layers(self):
        from catmap import events as evt
        from tests.test_events import DETAIL, SLD, _gj

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long\nA;40,5;2,5\nB;48,0;2,0\n".encode()

        app, st = self.load({"Navigation": "Temps réel", "ev_ct_csv_EMSR900": Up(), "button:Croiser": True})
        with self.fake_cyclones(), \
                mock.patch.object(evt, "fetch_activation_detail", return_value=(DETAIL, [])), \
                mock.patch.object(evt, "fetch_geojson", return_value=(_gj(), "")), \
                mock.patch.object(evt, "fetch_text", return_value=SLD):
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        _, res = st.session_state["ev_ct_cross_EMSR900"]
        self.assertIn("observedEventA.statut", res.columns)          # colonnes nommées d'après la couche
        self.assertEqual(list(res["observedEventA.statut"]), ["Dans la zone", "Hors zone"])
        self.assertIn("observedEventA.area_ha", res.columns)
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)

    def test_zone_tab_is_in_geolocalisation(self):
        app, st = self.load({"Navigation": "Géolocalisation"})
        with self.fake_cyclones():
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        tabs = [c for c in st.calls if c[0] == "tabs"]
        self.assertTrue(any("Entreprises dans une zone" in str(t) for t in tabs), tabs)

    def test_zone_activity_modes_render(self):
        for mode in ("Un secteur d'activité (code NAF)", "Une activité économique (langage naturel)"):
            app, st = self.load({"Navigation": "Géolocalisation", "zone_mode": mode})
            with self.fake_cyclones():
                app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
                app.main()
            self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)

    def test_georisques_page_renders(self):
        app, st = self.load({"Navigation": "Risque climatiques"})
        with self.fake_cyclones():
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertTrue(any(c[0] == "subheader" and "Géorisques" in str(c[1]) for c in st.calls), st.calls[:5])

    def test_fire_cross_with_csv(self):
        from catmap import events as evt
        from tests.test_events import _gj

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long\nA;40,5;2,5\nB;48,0;2,0\n".encode()

        app, st = self.load({"Navigation": "Temps réel", "ev_fire_csv": Up(), "button:Croiser": True})
        with self.fake_cyclones(), \
                mock.patch.object(evt, "burnt_areas", return_value=(_gj(), [])), \
                mock.patch.object(evt, "sample_layer", return_value=([45.0, 3.0], {
                    "mailles": 2, "valeurs_lues": 2, "objets": 2, "tronque": False, "hors_emprise": 0,
                    "presence": __import__("numpy").array([True, True]), "interroge": __import__("numpy").array([True, True]),
                    "statut": __import__("numpy").array(["valeur", "valeur"], dtype=object), "attributs": [None, None], "exemples": []})):
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        _, res, notes = st.session_state["ev_fire_cross"]
        self.assertEqual(list(res["incendie_statut"]), ["Dans une zone brûlée", "Non touché"])
        self.assertEqual(list(res["fwi_niveau"]), ["Très élevé", "Très faible"])
        self.assertIn("feu_mf010.fwi", res.columns)                       # colonnes liées aux couches sélectionnées
        self.assertIn("feu_mf010.fwi.statut", res.columns)
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertTrue([c for c in st.calls if c[0] == "download"])

    def test_flood_cross_with_csv(self):
        from catmap import events as evt

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long\nA;40,5;2,5\nB;48,0;2,0\n".encode()

        app, st = self.load({"Navigation": "Temps réel", "ev_flood_csv_efas": Up(), "button:Croiser": True,
                             "ev_flood_layers_efas": ["countries"]})
        with self.fake_cyclones(), \
                mock.patch.object(evt, "sample_layer", return_value=([2.5, float("nan")], {
                    "mailles": 2, "valeurs_lues": 1, "objets": 1, "tronque": False, "hors_emprise": 0,
                    "presence": __import__("numpy").array([True, False]), "interroge": __import__("numpy").array([True, True]),
                    "statut": __import__("numpy").array(["valeur", "aucun objet"], dtype=object), "attributs": [{"RP": "10"}, None], "exemples": []})):
            app.get_settings = lambda: __import__("catmap.config", fromlist=["x"]).load_settings(env={})
            app.main()
        _, res, notes, _ex = st.session_state["ev_flood_cross_efas"]
        self.assertEqual(list(res["crue_touche"]), ["Oui", "Non"])
        self.assertEqual(res.loc[0, "crue_countries"], 2.5)
        self.assertEqual(res.loc[0, "crue_countries.RP"], "10")
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

        app, st = self.load({"Navigation": "Temps réel", "vg_upload": Up()})
        with self.fake_cyclones(), mock.patch.dict("os.environ", {"METEOFRANCE_API_KEY": "K"}), \
                mock.patch.object(vg, "fetch_vigilance", return_value=PAYLOAD), \
                mock.patch.object(vg, "load_departements", return_value=vg.DepartementIndex(GEO)):
            app.main()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertTrue([c for c in st.calls if c[0] == "warning"])        # ligne sans coordonnées signalée
        metrics = {c[1][0]: c[1][1] for c in st.calls if c[0] == "metric"}
        self.assertEqual(metrics["Hors France (non traités)"], 2 - metrics["Sites en France"])
        self.assertFalse([c for c in st.calls if c[0] == "info"])           # plus de « lancez d'abord une géolocalisation »

    def test_vigilance_page_without_source_asks_for_file(self):
        from catmap import vigilance as vg
        from tests.test_vigilance import GEO, PAYLOAD
        app, st = self.load({"Navigation": "Temps réel"})
        with self.fake_cyclones(), mock.patch.dict("os.environ", {"METEOFRANCE_API_KEY": "K"}), \
                mock.patch.object(vg, "fetch_vigilance", return_value=PAYLOAD), \
                mock.patch.object(vg, "load_departements", return_value=vg.DepartementIndex(GEO)):
            app.main()
        infos = [c[1][0] for c in st.calls if c[0] == "info"]
        self.assertTrue(any("Chargez un fichier" in i for i in infos))

    def test_overseas_section_with_fake_payloads(self):
        from catmap import vigilance as vg
        from tests.test_vigilance import GEO, PAYLOAD
        app, st = self.load({"Navigation": "Temps réel"})
        def fake_path(auth, settings, path, session=None, binary=False):
            if binary:
                raise vg.VigilanceError("flux indisponible")
            return PAYLOAD
        with self.fake_cyclones(), mock.patch.dict("os.environ", {"METEOFRANCE_API_KEY": "K"}), \
                mock.patch.object(vg, "fetch_vigilance", return_value=PAYLOAD), \
                mock.patch.object(vg, "fetch_path", side_effect=fake_path), \
                mock.patch.object(vg, "load_departements", return_value=vg.DepartementIndex(GEO)):
            app.main()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        names = [c[1][0] for c in st.calls if c[0] == "metric"]
        self.assertIn("Polynésie française", names)
        self.assertIn("Nouvelle-Calédonie", names)

    def _depth_res(self):
        import pandas as pd
        return pd.DataFrame({"valeur": [1000.0, 2000.0, 500.0], "name": ["a", "b", "c"], "lat_f": [45.0, 45.1, 45.2], "lon_f": [4.0, 4.1, 4.2],
                             "debordement_RP10_raw": [0.0, 1.0, float("nan")], "debordement_RP100_raw": [1.0, 3.0, float("nan")]})

    def _jrc_upload(self):
        csv = ("continent,classe,hauteur_m,taux\nEurope,residential,0,0\nEurope,residential,1,0.2\nEurope,residential,3,0.6\n").encode()

        class Up:
            name = "jrc.csv"
            def getvalue(self): return csv
        return Up()

    def _hz(self, name="debordement", rps=(10, 100), result="fh_result"):
        return {"name": name, "label": name, "rps": list(rps), "col": (lambda r, n=name: f"{n}_RP{r}_raw"), "result": result}

    def test_damage_panel_runs_depths_first_then_damages_and_map(self):
        import pandas as pd
        res = self._depth_res()
        df = res[["valeur", "name"]].copy()
        app, st = self.load({"fh_jrc_up": [self._jrc_upload()], "button:Calculer les dommages": True, "fh_dmg_cont": "Europe", "fh_dmg_mode": "col", "fh_dmg_valcol": "valeur"})
        st.session_state["ex_dmg_open"] = True
        calls = []

        def ensure(names):                      # simule l'estimation des hauteurs : aucune hauteur en mémoire au départ
            calls.append(names)
            st.session_state["fh_result"] = ("pts.csv", len(df), res, {}, [10, 100], ["raw"])
        stored = app._fh_damage_panel(df, "pts.csv", [self._hz()], ensure)
        self.assertEqual(calls, [["debordement"]])
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        out = stored[2]["debordement"][0]
        self.assertAlmostEqual(out["dommage_debordement_RP100_eur"].iloc[0], 200.0)
        self.assertAlmostEqual(out["dommage_debordement_RP100_eur"].iloc[1], 1200.0)
        self.assertTrue(out["dommage_debordement_RP100_eur"].iloc[2] != out["dommage_debordement_RP100_eur"].iloc[2])      # NaN : pas de perte inventée
        from catmap import damage as dmg
        full = pd.concat([res, out], axis=1)
        kinds = {c: ("ratio" if c.endswith("_ratio") else "eur") for c in out.columns if c.startswith("dommage_debordement_RP") or c == "dommage_debordement_PAM_eur"}
        damage = {"kinds": kinds, "edges": {c: dmg.class_edges(out[c], k) for c, k in kinds.items()}}
        for col in ("dommage_debordement_RP100_eur", "dommage_debordement_RP100_ratio", "dommage_debordement_PAM_eur", "debordement_RP100_raw"):
            app._fh_map(full, {"raw": ["debordement_RP10_raw", "debordement_RP100_raw"]}, col, None, None, damage)
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)

    def test_damage_panel_does_not_rerun_depths_when_available(self):
        res = self._depth_res()
        df = res[["valeur", "name"]].copy()
        app, st = self.load({"fh_jrc_up": [self._jrc_upload()], "button:Calculer les dommages": True, "fh_dmg_cont": "Europe", "fh_dmg_mode": "col", "fh_dmg_valcol": "valeur"})
        st.session_state["ex_dmg_open"] = True
        st.session_state["fh_result"] = ("pts.csv", len(df), res, {}, [10, 100], ["raw"])
        calls = []
        stored = app._fh_damage_panel(df, "pts.csv", [self._hz()], lambda k: calls.append(k))
        self.assertEqual(calls, [])
        self.assertIsNotNone(stored)

    def test_damage_panel_per_point_continent_and_jrc_values_with_hazus(self):
        import pandas as pd
        curves = ("continent,classe,hauteur_m,taux\nEurope,industrial,0,0\nEurope,industrial,2,0.5\n"
                  "Asia,industrial,0,0\nAsia,industrial,2,0.2\n").encode()
        values = b"pays,classe,eur_m2\nFR,industrial,100\nVN,industrial,50\n"

        class Up:
            def __init__(self, name, data): self.name, self._d = name, data
            def getvalue(self): return self._d
        res = pd.DataFrame({"name": ["fr", "vn", "xx"], "surface_m2": [10.0, 10.0, 10.0], "pays_iso2": ["FR", "VN", "ZZ"],
                            "lat_f": [45.0, 10.0, 0.0], "lon_f": [4.0, 106.0, 0.0],
                            "debordement_RP100_raw": [2.0, 2.0, 2.0]})
        df = res[["name", "surface_m2", "pays_iso2"]].copy()
        script = {"fh_jrc_up": [Up("jrc_c.csv", curves), Up("jrc_v.csv", values)], "button:Calculer les dommages": True,
                  "fh_dmg_cls": "industrial", "fh_dmg_ctrycol": "pays_iso2", "fh_dmg_cont": "Europe", "fh_dmg_areacol": "surface_m2"}
        app, st = self.load(script)
        st.session_state["ex_dmg_open"] = True
        st.session_state["fh_result"] = ("p.csv", len(df), res, {}, [100], ["raw"])
        stored = app._fh_damage_panel(df, "p.csv", [self._hz(rps=[100])], lambda k: None)
        out = stored[2]["debordement"][0]
        self.assertEqual(list(out["dommage_continent_courbe"]), ["Europe", "Asia", "Europe"])      # Vietnam → Asie ; ZZ inconnu → défaut
        self.assertEqual(list(out["dommage_debordement_RP100_ratio"].round(3)), [0.5, 0.2, 0.5])
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        # valeurs JRC du pays (mode par défaut) avec l'approche Hazus
        from catmap import damage as dmg
        script["fh_dmg_method"] = "FEMA Hazus – fonctions riverines (États-Unis)"
        hz = dmg.parse_hazus(__import__("tests.test_damage", fromlist=["x"]).hazus_csv())
        app2, st2 = self.load(script)
        st2.session_state.update({"ex_dmg_open": True, "fh_hz": hz, "fh_result": ("p.csv", len(df), res, {}, [100], ["raw"])})
        stored2 = app2._fh_damage_panel(df, "p.csv", [self._hz(rps=[100])], lambda k: None)
        self.assertEqual(list(stored2[2]["debordement"][0]["dommage_valeur_exposee_eur"].round(1))[:2], [1000.0, 500.0])   # 10 m² × 100 / × 50 €/m²

    def test_damage_panel_two_hazards_share_parameters(self):
        import pandas as pd
        res_f = self._depth_res()
        res_c = pd.DataFrame({"valeur": [1000.0, 2000.0, 500.0], "name": ["a", "b", "c"], "lat_f": [45.0, 45.1, 45.2], "lon_f": [4.0, 4.1, 4.2],
                              "submersion_RP10_raw": [2.0, 0.0, 0.0], "submersion_RP100_raw": [3.0, 1.0, 0.0]})
        df = res_f[["valeur", "name"]].copy()
        app, st = self.load({"fh_jrc_up": [self._jrc_upload()], "button:Calculer les dommages": True, "fh_dmg_cont": "Europe",
                             "fh_dmg_mode": "col", "fh_dmg_valcol": "valeur"})
        st.session_state["ex_dmg_open"] = True
        st.session_state["fh_result"] = ("p.csv", 3, res_f, {}, [10, 100], ["raw"])
        st.session_state["cs_result"] = ("p.csv", 3, res_c, {}, [10, 100], ["raw"])
        hz = [self._hz(), self._hz("submersion", result="cs_result")]
        stored = app._fh_damage_panel(df, "p.csv", hz, lambda k: self.fail("hauteurs déjà calculées"))
        per = stored[2]
        self.assertEqual(set(per), {"debordement", "submersion"})
        self.assertAlmostEqual(per["debordement"][0]["dommage_debordement_RP100_eur"].iloc[0], 200.0)
        self.assertAlmostEqual(per["submersion"][0]["dommage_submersion_RP100_eur"].iloc[0], 600.0)     # 3 m → 60 % de 1000 €
        self.assertIn("dommage_submersion_PAM_eur", per["submersion"][0].columns)
        self.assertNotIn("dommage_debordement_RP100_eur", per["submersion"][0].columns)
        self.assertIn("ex_dmg_result", st.session_state)

    def test_damage_panel_warns_when_no_point_is_exposed(self):
        import pandas as pd
        res = pd.DataFrame({"valeur": [1000.0, 2000.0], "name": ["a", "b"], "lat_f": [45.0, 45.1], "lon_f": [4.0, 4.1],
                            "submersion_RP100_raw": [0.0, 0.0]})
        df = res[["valeur", "name"]].copy()
        app, st = self.load({"fh_jrc_up": [self._jrc_upload()], "button:Calculer les dommages": True, "fh_dmg_cont": "Europe",
                             "fh_dmg_mode": "col", "fh_dmg_valcol": "valeur"})
        st.session_state["ex_dmg_open"] = True
        st.session_state["cs_result"] = ("p.csv", 2, res, {}, [100], ["raw"])
        stored = app._fh_damage_panel(df, "p.csv", [self._hz("submersion", [100], "cs_result")], lambda k: None)
        self.assertEqual(stored[2]["submersion"][1]["points_exposes"], 0)
        self.assertTrue([c for c in st.calls if c[0] == "warning" and "aucun point" in c[1][0]], st.calls)

    def test_exposition_section_multi_hazard_end_to_end(self):
        import pandas as pd
        from catmap import coastalhazard as ch
        from catmap import floodhazard as fh

        class Up:
            name = "pts.csv"
            def getvalue(self): return b"name;lat;long\nA;43,3;5,4\nB;48,8;2,3\n"

        calls = []

        def fake_coast(df, lat_c, lon_c, rps, scen, sub, year, **k):
            calls.append(("coast", list(rps)))
            out = df.copy().reset_index(drop=True)
            for r in rps:
                out[ch.column_name(r)] = [1.5, 0.0]
            out["submersion_statut"] = "ok"
            return out, {"points": 2, "invalides": 0, "fichiers": len(rps), "fichiers_lus": {}, "problemes": {}}

        def fake_flood(df, lat_c, lon_c, rps, kinds, catalog, **k):
            calls.append(("flood", list(rps), list(kinds)))
            out = df.copy().reset_index(drop=True)
            for r in rps:
                for kd in kinds:
                    out[fh.column_name(r, kd)] = [0.8, 0.0]
            out["debordement_tuile"] = ["ID1", None]
            out["debordement_statut"] = ["ok", "hors tuile (mer ou zone non couverte)"]
            return out, {"points": 2, "hors_tuile": 1, "tuiles": 1, "fichiers": 1, "problemes": {}, "tuiles_detail": [], "fichiers_lus": {}}
        app, st = self.load({"ex_hazards": ["debordement", "submersion"], "ex_csv_0": Up(), "button:Estimer la hauteur d'eau": True,
                             "ex_rps": [50, 100]})
        with mock.patch.object(ch, "estimate_depths_ee", side_effect=fake_coast), mock.patch.object(fh, "estimate_depths", side_effect=fake_flood), \
                mock.patch.object(app, "_fh_catalog", return_value=object()), mock.patch.object(fh, "plan", return_value={"tile_ids": [1]}):
            app._exposition_section()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertEqual(sorted(c[0] for c in calls), ["coast", "flood"])                       # les deux aléas lancés d'un coup
        self.assertEqual([c for c in calls if c[0] == "flood"][0][1], [50, 100])                # RP commun appliqué aux deux
        self.assertEqual([c for c in calls if c[0] == "coast"][0][1], [50, 100])
        self.assertIn("fh_result", st.session_state)
        self.assertIn("cs_result", st.session_state)
        self.assertTrue([c for c in st.calls if c[0] == "download" and c[1].startswith("Télécharger")])

    def test_submersion_cells_option_draws_pixels(self):
        from catmap import coastalhazard as ch

        class Up:
            name = "pts.csv"
            def getvalue(self): return b"name;lat;long\nA;43,3;5,4\n"

        def fake_coast(df, lat_c, lon_c, rps, *a, **k):
            out = df.copy().reset_index(drop=True)
            for r in rps:
                out[ch.column_name(r)] = [1.0]
            out["submersion_statut"] = "ok"
            return out, {"points": 1, "invalides": 0, "fichiers": len(rps), "fichiers_lus": {ch.column_name(100): "IMG100"}, "problemes": {}}
        app, st = self.load({"ex_hazards": ["submersion"], "ex_csv_0": Up(), "button:Estimer la hauteur d'eau": True, "ex_rps": [100],
                             "ex_show_cells": True, "ex_color": "submersion_RP100_raw"})
        got = {}
        st.checkbox = lambda label, value=False, **k: True if k.get("key") == "ex_show_cells" else value   # la fausse case ignore le script
        with mock.patch.object(ch, "estimate_depths_ee", side_effect=fake_coast), \
                mock.patch.object(ch, "flood_cells", side_effect=lambda i, lo, la, r: got.update(i=i, r=r) or [(5.4, 43.3, 1.2)]), \
                mock.patch.object(app, "_fh_map") as fm:
            app._exposition_section()
        self.assertEqual((got["i"], got["r"]), ("IMG100", 5.0))
        self.assertEqual(fm.call_args[0][-1], [(5.4, 43.3, 1.2)])
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)

    def test_exposition_section_common_rp_skips_unavailable_for_a_hazard(self):
        from catmap import coastalhazard as ch

        class Up:
            name = "pts.csv"
            def getvalue(self): return b"name;lat;long\nA;43,3;5,4\n"
        got = {}

        def fake_coast(df, lat_c, lon_c, rps, *a, **k):
            got["rps"] = list(rps)
            out = df.copy().reset_index(drop=True)
            for r in rps:
                out[ch.column_name(r)] = [1.0]
            out["submersion_statut"] = "ok"
            return out, {"points": 1, "invalides": 0, "fichiers": len(rps), "fichiers_lus": {}, "problemes": {}}
        app, st = self.load({"ex_hazards": ["submersion"], "ex_csv_0": Up(), "button:Estimer la hauteur d'eau": True, "ex_rps": [75, 100, 1000]})
        with mock.patch.object(ch, "estimate_depths_ee", side_effect=fake_coast):
            app._exposition_section()
        self.assertEqual(got["rps"], [100, 1000])                                              # RP75 n'existe pas pour la submersion

    def test_wind_tab_reads_catalogue_without_key(self):
        from catmap import copernicus as cop
        form = [{"name": "variable", "label": "V", "type": "T", "details": {"values": ["fg10_gust", "t2m"]}}]
        app, st = self.load({"button:Lire le catalogue Copernicus": True})
        with mock.patch.object(cop, "fetch_form", return_value=form):
            app._wind_section()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertEqual(set(st.session_state["cop_forms"]), {"timeseries", "storms"})
        self.assertFalse([c for c in st.calls if c[0] == "info"])

    def test_wind_end_to_end_with_fake_cds(self):
        from catmap import wind as w
        from tests.test_wind import FakeClient

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long;surface\nA;48,0;2,0;1000\nB;48,1;2,1;2000\n".encode()

        with tempfile.TemporaryDirectory() as t:
            cache = Path(t)
            app, st = self.load({"wd_csv_0": Up(), "wd_sel": ["vent", "tempete"], "wd_rps": [10, 100], "wd_y0": 2000, "wd_y1": 2000,
                                 "button:Lancer / reprendre le calcul": True, "button:Estimer les dommages": True,
                                 "wd_mode": "area", "wd_unit": 1000.0, "button:Calculer les dommages": True})
            st.session_state["wd_dmg_open"] = True
            app._secrets = lambda: {"CDSAPI_KEY": "k"}
            with mock.patch.object(w, "make_client", return_value=FakeClient()), mock.patch.object(w, "fetch_tracks", return_value=w.parse_tracks(
                        b"track_id,time,lon,lat,gust\ns1,2000-01-02 00:00,2,48,30\ns1,2000-01-02 06:00,3,49,35\n")):
                app._wind_section()
        w.clear_cache()
        self.assertFalse([c for c in st.calls if c[0] == "error"], [c for c in st.calls if c[0] == "error"])
        res = st.session_state["wd_result"]
        self.assertEqual(res["info"]["done"], res["info"]["cells"])
        self.assertIn("vent_cellule_era5", res["df"].columns)
        self.assertIsNotNone(res["storms"], res["info"])
        self.assertEqual(len(res["storms"]), 1)

    def test_seisme_end_to_end_and_tab_order(self):
        from catmap import seismic as smc
        from tests.test_seismic import pga_tif

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long;surface\nA;59,5;20,5;1000\nB;55,0;25,0;2000\n".encode()

        with tempfile.TemporaryDirectory() as t:
            pga_tif(Path(t) / "v2023_1_pga_475_rock_3min.tif")
            app, st = self.load({"sm_csv_0": Up(), "button:Estimer l'aléa sismique": True, "button:Estimer les dommages": True,
                                 "button:Calculer les dommages": True, "sm_mode": "area", "sm_unit": 1000.0, "sm_area": "surface"})
            st.session_state["sm_dmg_open"] = True
            with mock.patch.object(smc, "gem_path", return_value=Path(t) / "v2023_1_pga_475_rock_3min.tif"), \
                    mock.patch.object(smc, "compact_path", return_value=None):
                app._seisme_section()
        self.assertFalse([c for c in st.calls if c[0] == "error"], [c for c in st.calls if c[0] == "error"])
        res = st.session_state["sm_result"]
        self.assertEqual(res["summ"]["ok"], 2)
        dm = st.session_state["sm_dmg"]
        self.assertIn("dommage_seisme_RP475_eur", dm["out"].columns)
        r = dm["out"]["dommage_seisme_RP475_ratio"]
        self.assertGreater(r.iloc[0], r.iloc[1])                       # PGA 0,4 g > 0,05 g

    def test_seisme_compact_and_raw_checkbox(self):
        from catmap import seismic as smc
        from tests.test_seismic import pga_tif

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long;surface\nA;59,5;20,5;1000\nB;55,0;25,0;2000\n".encode()

        for raw, source in ((False, "fichier précalculé"), (True, "raster GEM brut")):
            with tempfile.TemporaryDirectory() as t:
                tif = Path(t) / "v2023_1_pga_475_rock_3min.tif"
                pga_tif(tif)
                comp = Path(t) / smc.COMPACT_NAME
                smc.build_compact(tif, comp, progress=lambda *_: None)
                app, st = self.load({"sm_csv_0": Up(), "button:Estimer l'aléa sismique": True})
                app.st.checkbox = lambda label, value=False, **k: raw if k.get("key") == "sm_raw" else value
                with mock.patch.object(smc, "gem_path", return_value=tif), mock.patch.object(smc, "compact_path", return_value=comp):
                    app._seisme_section()
            self.assertFalse([c for c in st.calls if c[0] == "error"], [c for c in st.calls if c[0] == "error"])
            res = st.session_state["sm_result"]
            self.assertEqual(res["summ"]["source"], source)
            self.assertEqual(res["summ"]["ok"], 2)

    def test_seisme_eshm20_flow_with_fake_service(self):
        from catmap import seismic as smc

        class Up:
            name = "pts.csv"
            def getvalue(self): return b"nom;lat;long;surface\nA;46,2;6,15;1000\nB;46,21;6,16;1000\n"

        def fake_get(path, params, session=None, timeout=40):
            if path == "/v1/models":
                return [{"id": 7, "name": "ESHM20"}]
            return {"imls": [0.01, 0.05, 0.1, 0.2, 0.4, 0.8], "poes": [0.2, 0.03, 0.008, 0.0021, 0.0004, 0.00005]}

        with tempfile.TemporaryDirectory() as t:
            app, st = self.load({"sm_csv_0": Up(), "sm_ds": "eshm20", "button:Estimer l'aléa sismique": True})
            with mock.patch.object(smc, "_get_json", side_effect=fake_get), mock.patch.object(smc, "CACHE_DIR", Path(t)):
                with mock.patch.object(smc, "estimate_pga", wraps=lambda *a, **k: smc._estimate_eshm(a[0], a[1], a[2], k.get("rps", a[4] if len(a) > 4 else []), None, Path(t), 50)):
                    app._seisme_section()
        self.assertFalse([c for c in st.calls if c[0] == "error"], [c for c in st.calls if c[0] == "error"])
        res = st.session_state["sm_result"]
        self.assertIn("seisme_RP475_pga", res["df"].columns)
        self.assertIn("seisme_RP2475_pga", res["df"].columns)
        self.assertAlmostEqual(res["df"]["seisme_RP475_pga"].iloc[0], 0.2, delta=0.02)

    def _hail_env(self, t):
        """Remplace les accès Zenodo / cache de catmap.hail par des fichiers NetCDF locaux (aucun accès réseau)."""
        from catmap import hail as hlc
        from tests.test_hail import FakeZenodo, build_source
        src, cache, nodata = Path(t) / "src", Path(t) / "cache", Path(t) / "nodata"
        build_source(src)
        sess = FakeZenodo(src)
        orig_est, orig_ser, orig_cells = hlc.estimate, hlc.load_series, hlc.series_cache_cells
        return [mock.patch.object(hlc, "estimate", lambda *a, **k: orig_est(*a, session=sess, cache_dir=cache, data_dir=nodata, **k)),
                mock.patch.object(hlc, "load_series", lambda cells, years=hlc.YEARS, cache_dir=cache: orig_ser(cells, years, cache_dir)),
                mock.patch.object(hlc, "series_cache_cells", lambda cache_dir=cache, years=hlc.YEARS: orig_cells(cache_dir, years))]

    def test_grele_climatology_end_to_end(self):
        import contextlib

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long;valeur\nA;45,0;5,0;1000000\nB;45,0;0,25;2000000\n".encode()

        with tempfile.TemporaryDirectory() as t:
            app, st = self.load({"gr_csv_0": Up(), "gr_mode": "clim", "button:Estimer l'aléa grêle": True, "button:Estimer les dommages": True,
                                 "button:Calculer les dommages": True, "gr_vmode": "col", "gr_valcol": "valeur"})
            st.session_state["gr_dmg_open"] = True
            with contextlib.ExitStack() as stack:
                for m in self._hail_env(t):
                    stack.enter_context(m)
                app._grele_section()
        self.assertFalse([c for c in st.calls if c[0] == "error"], [c for c in st.calls if c[0] == "error"])
        res = st.session_state["gr_result"]
        self.assertEqual(res["summ"]["ok"], 2)
        self.assertAlmostEqual(res["df"]["grele_moy_freq"].iloc[0], 0.1)
        o = st.session_state["gr_dmg"]["out"]
        self.assertIn("dommage_grele_PAM_eur", o.columns)
        self.assertGreater(o["dommage_grele_PAM_eur"].iloc[0], 0)
        self.assertEqual(o["dommage_grele_PAM_eur"].iloc[1], 0)         # clim nulle à lon 0,25

    def test_grele_damage_method_2_schmid(self):
        import contextlib

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long;valeur\nA;45,0;5,0;1000000\nB;45,0;0,25;2000000\n".encode()

        with tempfile.TemporaryDirectory() as t:
            app, st = self.load({"gr_csv_0": Up(), "gr_mode": "clim", "button:Estimer l'aléa grêle": True, "button:Estimer les dommages": True,
                                 "button:Calculer les dommages": True, "gr_vmode": "col", "gr_valcol": "valeur", "gr_ratio_mode": "schmid"})
            st.session_state["gr_dmg_open"] = True
            with contextlib.ExitStack() as stack:
                for m in self._hail_env(t):
                    stack.enter_context(m)
                app._grele_section()
        self.assertFalse([c for c in st.calls if c[0] == "error"], [c for c in st.calls if c[0] == "error"])
        o = st.session_state["gr_dmg"]["out"]
        from catmap import hail as hl
        self.assertAlmostEqual(o["dommage_grele_PAM_eur"].iloc[0], 0.1 * hl.schmid_mdr(50, "buildings") * 1_000_000, places=4)
        self.assertIn("Schmid et al. 2024", o["dommage_methode"].iloc[0])

    def test_grele_annual_end_to_end_with_viewer_and_trend(self):
        import contextlib
        if importlib.util.find_spec("altair") is None:
            sys.modules["altair"] = mock.MagicMock()
            self.addCleanup(sys.modules.pop, "altair", None)

        class Up:
            name = "pts.csv"
            def getvalue(self): return "nom;lat;long;valeur\nA;45,0;2,0;1000000\nB;45,0;1,0;2000000\n".encode()

        with tempfile.TemporaryDirectory() as t:
            app, st = self.load({"gr_csv_0": Up(), "gr_mode": "annual", "button:Estimer l'aléa grêle": True, "button:Estimer les dommages": True,
                                 "button:Calculer les dommages": True, "button:Visualiser les séries annuelles": True,
                                 "gr_vmode": "col", "gr_valcol": "valeur"})
            st.session_state["gr_dmg_open"] = True
            app.st.checkbox = lambda label, value=False, **k: True if k.get("key") == "gr_trend" else value
            with contextlib.ExitStack() as stack:
                for m in self._hail_env(t):
                    stack.enter_context(m)
                app._grele_section()
        self.assertFalse([c for c in st.calls if c[0] == "error"], [c for c in st.calls if c[0] == "error"])
        res = st.session_state["gr_result"]
        self.assertEqual(res["summ"]["en_attente"], 0)
        self.assertEqual(res["summ"]["ok"], 2)
        self.assertIn("grele_RP50_freq_brut", res["df"].columns)
        self.assertEqual(res["df"]["grele_facteur_tendance"].iloc[0], 2.0)    # lon 2 : hausse significative, plafonnée
        self.assertEqual(res["df"]["grele_facteur_tendance"].iloc[1], 1.0)    # lon 1 : p = 0,5
        o = st.session_state["gr_dmg"]["out"]
        self.assertIn("dommage_grele_RP50_eur", o.columns)
        self.assertIn("dommage_grele_PAM_eur", o.columns)
        self.assertTrue([c for c in st.calls if c[0] == "altair_chart"])

    def test_runoff_section_end_to_end(self):
        from catmap import runoff as rs

        class Up:
            name = "pts.csv"
            def getvalue(self): return b"name;lat;long\nA;43,62;7,10\nB;40,0;-30,0\n"

        def fake_estimate(df, lat_c, lon_c, radius, ee_p, ee_a, on_progress=None, **k):
            out = df.copy().reset_index(drop=True)
            for c in list(rs.COLUMNS.values()) + list(rs.RAW.values()):
                out[c] = [0.5, None]
            out[rs.COLUMNS["classe"]] = [4, None]
            out[rs.STATUS] = ["ok", "hors terre ou hors couverture"]
            if on_progress:
                on_progress(1, 1)
            return out, {"points": 2, "invalides": 0, "pixels_uniques": 2, "lots": 1, "echecs": 0, "hors_terre": 1, "facteurs": ["slope"],
                         "jeux_absents": ["gcn"], "rayon_km": radius, "repartition": {"forte": 1}}
        app, st = self.load({"rs_csv_0": Up(), "button:Estimer la susceptibilité": True})
        with mock.patch.object(rs, "estimate", side_effect=fake_estimate), mock.patch.object(app, "_ev_show_map") as fm:
            app._ruissellement_section()
        self.assertFalse([c for c in st.calls if c[0] == "error"], st.calls)
        self.assertIn("rs_result", st.session_state)
        self.assertTrue(fm.called)
        self.assertTrue([c for c in st.calls if c[0] == "warning"])          # jeu absent signalé
        self.assertTrue([c for c in st.calls if c[0] == "download"])

    def test_grele_tab_between_seisme_and_georisques(self):
        app, st = self.load({})
        app.page_risques_climatiques()
        labels = [c[1] for c in st.calls if c[0] == "tabs"][0]
        self.assertEqual(labels, ["Débordement et submersion", "Ruissellement pluvial", "Vent & tempête", "Séisme", "Grêle", "Géorisques"])

    def test_wind_fit_viewer(self):
        import pandas as pd
        import numpy as np
        if importlib.util.find_spec("altair") is None:                  # altair vient avec Streamlit ; ici on le simule pour exécuter le code de tracé
            sys.modules["altair"] = mock.MagicMock()
            self.addCleanup(sys.modules.pop, "altair", None)
        rng = np.random.default_rng(1)
        cells = [(48.0, 2.0), (45.0, 5.0)]
        fd = {c: {"annual": pd.Series(rng.gumbel(25, 3, 40), index=range(1984, 2024)), "peaks": 20 + rng.exponential(4, 200)} for c in cells}
        res = {"name": "pts", "n": 2, "sel": ["vent", "tempete"], "cells": cells, "fitdata": fd, "n_years": 40,
               "storms": pd.DataFrame({"storm": ["a"]}), "df": pd.DataFrame({"name": ["A", "B"]})}
        for kind in ("vent", "tempete"):
            app, st = self.load({"wd_fit_kind": kind, "button:Visualiser les lois ajustées": True})
            st.session_state.pop("wd_fit_open", None)
            app._wind_fit_viewer(res, "hodges")
            self.assertFalse([c for c in st.calls if c[0] == "error"])
            self.assertTrue([c for c in st.calls if c[0] == "altair_chart"], kind)
            self.assertTrue(isinstance(st.session_state["wd_fit_table"][1], pd.DataFrame))

    def test_auto_resume_logic(self):
        app, st = self.load({})
        with mock.patch("time.sleep") as sl:
            self.assertTrue(app._auto_resume("t", 5, 3))                       # reste du travail : relance demandée
            self.assertEqual([c[0] for c in st.calls].count("rerun"), 1)
            self.assertTrue(st.session_state["t_job"])
            self.assertTrue(app._auto_resume("t", 4, 4, queue_full=True))      # progrès + file pleine : attente puis relance
            self.assertEqual(sl.call_args_list[-1][0][0], 45)
            for _ in range(5):                                                 # plus aucun progrès : arrêt après max_stall essais
                last = app._auto_resume("t", 4, 4, max_stall=3)
            self.assertFalse(last)
            self.assertFalse(st.session_state["t_job"])
            self.assertFalse(app._auto_resume("t", 0, 9))                      # terminé : pas de relance
            self.assertFalse(st.session_state["t_job"])

    def test_wind_tab_reports_unreachable_catalogue(self):
        from catmap import copernicus as cop
        app, st = self.load({"button:Lire le catalogue Copernicus": True})
        with mock.patch.object(cop, "fetch_form", side_effect=cop.CopernicusError("HTTP 503")):
            app._wind_section()
        self.assertTrue([c for c in st.calls if c[0] == "error" and "503" in c[1][0]])

    def test_secrets_flattened_and_settings_not_stale(self):
        app, st = self.load({"Navigation": "Temps réel"})
        flat = app._flatten_secrets({"A": "1", "section": {"METEOFRANCE_API_KEY": "k", "sub": {"X": "y"}}}, {})
        self.assertEqual(flat, {"A": "1", "METEOFRANCE_API_KEY": "k", "X": "y"})
        s1 = app.get_settings()
        self.assertEqual(s1.area_max_radius_m, 10000)
        self.assertEqual(s1.area_max_sites, 50000)

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
