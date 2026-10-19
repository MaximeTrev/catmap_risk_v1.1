"""CatMap Risk - application Streamlit.

Lancer :  streamlit run app.py

Toute la logique métier est dans le paquet ``catmap`` (testé sans Streamlit) ; ce fichier ne
contient que l'interface. Points clés par rapport à la v1 :
- la recherche part d'un bouton « Lancer » et le résultat est mémorisé (``st.session_state``) :
  changer un filtre ne relance plus jamais Overpass (A20) ;
- un résultat vide ou une erreur n'est plus une exception (A6) ;
- tous les textes issus d'OSM sont échappés avant d'entrer dans une popup HTML ;
- carte avec regroupement (MarkerCluster) et plafond d'affichage (A13) ;
- chemins basés sur ``__file__`` : lancement possible depuis n'importe quel dossier.
"""
from __future__ import annotations

import html
import io
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from catmap.analysis import completeness, summary_by_company, top_with_other
from catmap.countries import get_country_index
from catmap.inputs import InputError, parse_names_text, read_company_list
from catmap.models import NameQuery, SearchOptions
from catmap.naf import NafClassifier
from catmap.overpass import OverpassClient
from catmap.pipeline import geolocate
from catmap.config import load_settings
from catmap import vigilance as vg

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MAX_MARKERS = 5000
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
DISPLAY_COLUMNS = ["nom_demande", "name", "pays", "naf_section", "naf_division", "surface_m2", "match_score",
                   "match_field", "geom_kind", "lat", "long", "naf_source", "naf_confidence", "addr_street",
                   "addr_postcode", "addr_city", "brand", "osm_url"]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# --------------------------------------------------------------------------
# Ressources partagées (chargées une seule fois par processus)
# --------------------------------------------------------------------------
def _secrets() -> dict:
    try:
        return {k: str(v) for k, v in st.secrets.items() if not isinstance(v, dict)}
    except Exception:  # noqa: BLE001 - pas de fichier secrets.toml : on s'appuie sur l'environnement
        return {}


@st.cache_resource
def get_settings():
    return load_settings(secrets=_secrets())


@st.cache_resource
def get_client():
    return OverpassClient(get_settings())


@st.cache_resource
def get_countries():
    return get_country_index()


@st.cache_resource
def get_naf():
    return NafClassifier()


@st.cache_resource
def get_departements():
    return vg.load_departements(get_settings())


# --------------------------------------------------------------------------
# Aides d'affichage (compatibles avec plusieurs versions de Streamlit)
# --------------------------------------------------------------------------
def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig")


def _xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def _popup_html(row: pd.Series) -> str:
    def esc(v):
        return "" if pd.isna(v) else html.escape(str(v))
    surf = "" if pd.isna(row.get("surface_m2")) else f"<br>Surface : {row['surface_m2']:,.0f} m²".replace(",", " ")
    return (f"<div style='font:12px sans-serif;min-width:180px'><b>{esc(row['name'])}</b><br>"
            f"{esc(row['nom_demande'])}<br>{esc(row.get('pays'))}<br>{esc(row.get('naf_section'))}{surf}<br>"
            f"Score : {esc(row.get('match_score'))}<br>"
            f"<a href='{esc(row['osm_url'])}' target='_blank'>Voir sur OpenStreetMap</a></div>")


_COLOR_CHOICES = {"Entreprise": "nom_demande", "Pays": "pays", "Secteur NAF": "naf_section", "Aucune (une seule couleur)": None}


def _site_map(df: pd.DataFrame):
    """Carte interactive des sites du tableau : fonds de carte au choix, regroupement, popups, plein écran."""
    import folium
    from folium.plugins import Fullscreen, MarkerCluster
    from streamlit_folium import st_folium

    d = df.dropna(subset=["lat", "long"])
    if d.empty:
        st.info("Aucun site à afficher.")
        return
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Colorer les points par", list(_COLOR_CHOICES), key="map_color")
    cluster_on = c2.checkbox("Regrouper les points proches", True, key="map_cluster")
    color_col = _COLOR_CHOICES[choice]

    if len(d) > MAX_MARKERS:
        st.warning(f"Carte limitée à {MAX_MARKERS} sites sur {len(d)} (les mieux appariés). "
                   "Filtrez par entreprise ou par pays pour tout voir.")
        d = d.sort_values("match_score", ascending=False).head(MAX_MARKERS)

    m = folium.Map(location=[float(d["lat"].mean()), float(d["long"].mean())], zoom_start=5,
                   control_scale=True, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Plan (OpenStreetMap)").add_to(m)
    folium.TileLayer("CartoDB positron", name="Fond clair").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Fond sombre").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
    Fullscreen().add_to(m)
    m.fit_bounds([[float(d["lat"].min()), float(d["long"].min())], [float(d["lat"].max()), float(d["long"].max())]])

    if color_col:
        keys = sorted(d[color_col].fillna("n/d").astype(str).unique())
        colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}
    else:
        keys, colors = [], {}
    layer = MarkerCluster(name="Sites").add_to(m) if cluster_on else folium.FeatureGroup(name="Sites").add_to(m)
    for _, r in d.iterrows():
        col = colors.get(str(r[color_col]) if pd.notna(r[color_col]) else "n/d", "#1f77b4") if color_col else "#bd8e43"
        folium.CircleMarker(
            [float(r["lat"]), float(r["long"])], radius=6, color=col, fill=True, fill_opacity=0.75,
            popup=folium.Popup(_popup_html(r), max_width=320), tooltip=html.escape(str(r["name"])),
        ).add_to(layer)
    folium.LayerControl(collapsed=True).add_to(m)

    if keys:
        shown = keys[:12]
        st.markdown(" &nbsp; ".join(
            f"<span style='color:#000'><span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
            f"background:{colors[k]};margin-right:4px'></span>{html.escape(k)}</span>" for k in shown)
            + (" …" if len(keys) > len(shown) else ""), unsafe_allow_html=True)
    key = f"map_{color_col}_{cluster_on}_{len(d)}_{round(float(d['lat'].sum()), 4)}"
    try:
        st_folium(m, height=620, returned_objects=[], key=key, use_container_width=True)
    except TypeError:  # anciennes versions de streamlit-folium
        st_folium(m, height=620, width=1400, returned_objects=[], key=key)
    st.caption(f"{len(d)} site(s) affiché(s) : ceux du tableau ci-dessus (filtres Entreprises / Pays appliqués). "
               "Cliquez sur un point pour le détail ; le sélecteur en haut à droite change le fond de carte.")


def _pie(df: pd.DataFrame, col: str, title: str):
    import plotly.express as px
    t = top_with_other(df[col], top_n=10)
    if t.empty:
        st.info(f"Aucune donnée « {title.lower()} » renseignée pour cette sélection.")
        return
    fig = px.pie(t, names="modalite", values="n", hole=0.3, title=title)
    fig.update_traces(textinfo="percent", texttemplate="%{percent:.0%}")
    _stretch(st.plotly_chart, fig)


# --------------------------------------------------------------------------
# Page « Géolocalisation »
# --------------------------------------------------------------------------
def _collect_queries(mode: str) -> tuple[list[NameQuery], list[str]]:
    if mode == "Un nom d'entreprise":
        name = st.text_input("Nom de l'entreprise", placeholder="Ex. IKEA", key="in_name")
        qid = st.text_input("Identifiant Wikidata (facultatif, recherche très précise)", placeholder="Ex. Q54078", key="in_qid")
        if qid.strip():
            st.info(f"Identifiant Wikidata renseigné ({qid.strip().upper()}) : il prime sur le nom et peut ramener une autre "
                    "entreprise que celle saisie. Videz ce champ pour chercher uniquement par nom.")
        return ([NameQuery.make(name, qid)] if name.strip() else []), []
    if mode == "Liste (une entreprise par ligne)":
        txt = st.text_area("Une entreprise par ligne ; identifiant Wikidata facultatif après « ; »",
                           placeholder="IKEA;Q54078\nLeroy Merlin", height=140, key="in_text")
        if not txt.strip():
            return [], []
        try:
            return parse_names_text(txt)
        except InputError as e:
            st.error(str(e))
            return [], []
    up = st.file_uploader("Fichier CSV ou Excel (nom d'entreprise en première colonne)", type=["csv", "txt", "xlsx"], key="in_file")
    header = st.radio("Première ligne", ["Détecter automatiquement", "C'est un en-tête", "C'est une entreprise"],
                      horizontal=True, key="in_header")
    if up is None:
        return [], []
    flag = {"Détecter automatiquement": None, "C'est un en-tête": True, "C'est une entreprise": False}[header]
    try:
        qs, warns = read_company_list(up.getvalue(), up.name, flag)
    except InputError as e:
        st.error(str(e))
        return [], []
    st.caption(f"{len(qs)} entreprise(s) lue(s).")
    return qs, warns


def _options_ui() -> SearchOptions:
    s = get_settings()
    with st.expander("Options de recherche (compromis rappel / précision)"):
        c1, c2 = st.columns(2)
        with c1:
            strict = st.checkbox("Nom en mot entier (recommandé)", True, help="Écarte « xxIKEAxx » ; désactiver augmente le bruit.")
            biz = st.checkbox("Établissements uniquement", True, help="Exige un tag shop, amenity, office, craft… (écarte arrêts, monuments, etc.).")
            rel = st.checkbox("Inclure les grands sites (relations)", True)
            op = st.checkbox("Chercher aussi dans « operator »", False, help="Plus de résultats, plus de bruit.")
        with c2:
            idx = get_countries()
            labels = sorted(f"{a['name_fr']} ({a['iso2']})" for a in idx.attrs if a.get("iso2") and a.get("name_fr"))
            picked = st.multiselect("Limiter à des pays (recommandé pour un grand périmètre)", labels)
            min_score = st.slider("Score minimal conservé", 0.0, 1.0, 0.0, 0.05,
                                  help="1,0 = identifiant Wikidata ; 0,9 = nom identique ; 0,75 = commence par ; 0,6 = contenu.")
            dedup = st.slider("Fusion des doublons node/way (mètres)", 0, 100, 25)
            fast = st.checkbox("Mode rapide (noms exacts uniquement)", False,
                               help="Interroge des graphies exactes (indexées) au lieu d'une regex : beaucoup plus rapide, "
                                    "mais ne trouve pas « IKEA Paris Nord » si le nom n'est pas exactement « IKEA ».")
    if s.uses_public_overpass:
        st.caption("Serveurs Overpass **publics** : usage raisonnable demandé. "
                   + ("" if s.contact else "Renseignez `CATMAP_CONTACT` (e-mail ou URL) pour vous identifier. ")
                   + "Pour de gros volumes, voir « Overpass local » dans le README.")
    else:
        st.caption(f"Instance Overpass **locale** : {s.overpass_urls[0]}")
    return SearchOptions(include_operator=op, include_relations=rel, require_business_tag=biz,
                         countries=tuple(p.rsplit("(", 1)[1].rstrip(")") for p in picked),
                         strict_word_match=strict, min_score=min_score, dedup_radius_m=float(dedup),
                         search_mode="exact" if fast else "regex")


def _run_search(queries: list[NameQuery], options: SearchOptions):
    box = st.empty()
    bar = box.progress(0.0, text="Démarrage…")

    def cb(done: int, total: int, name: str):
        bar.progress(done / total, text=f"{done}/{total} — {name}")

    status = st.empty()
    res = geolocate(queries, options, get_client(), get_settings(), get_countries(), get_naf(), progress=cb,
                    on_event=(lambda m: status.caption(m)) if get_settings().max_workers == 1 else None)
    status.empty()
    box.empty()
    st.session_state["result"] = res
    st.session_state["result_options"] = options
    st.session_state["result_queries"] = [(q.raw.casefold(), q.wikidata) for q in queries]


def _status_messages(report: pd.DataFrame):
    for statut, fn, label in (("erreur", st.error, "en erreur"), ("partiel", st.warning, "avec résultat partiel"),
                              ("aucun résultat", st.info, "sans résultat")):
        sub = report[report["statut"] == statut]
        if sub.empty:
            continue
        lines = [f"- **{r.nom_demande}**" + (f" : {r.message}" if r.message else "") for r in sub.itertuples()]
        fn(f"{len(sub)} entreprise(s) {label} :\n\n" + "\n".join(lines[:30]))


def _render_result(res):
    sites, report = res.sites, res.report
    n_req = len(report)
    n_found = int((report["statut"].isin(["ok", "partiel"]) & (report["n_sites"] > 0)).sum())
    comp = completeness(sites)
    c = st.columns(5)
    c[0].metric("Sites", f"{len(sites):,}".replace(",", " "))
    c[1].metric("Entreprises trouvées", f"{n_found} / {n_req}")
    c[2].metric("Pays renseigné", f"{comp['pays']:.0f} %")
    c[3].metric("Secteur NAF renseigné", f"{comp['naf_section']:.0f} %")
    c[4].metric("Base OSM", (res.meta["snapshots"][0][:10] if res.meta["snapshots"] else "n/d"),
                help="Date de la base Overpass interrogée (traçabilité).")
    _status_messages(report)
    if sites.empty:
        st.info("Aucun site à afficher. Essayez de désactiver « Nom en mot entier » ou « Établissements uniquement », "
                "ou de renseigner un identifiant Wikidata.")
        _render_report_tab(res)
        return

    f1, f2 = st.columns(2)
    companies = sorted(sites["nom_demande"].unique())
    pick_c = f1.multiselect("Entreprises", companies, default=companies, key="f_company")
    countries = sorted(sites["pays"].dropna().unique())
    pick_p = f2.multiselect("Pays", countries, key="f_country", help="Vide = tous (y compris sans pays)")
    view = sites[sites["nom_demande"].isin(pick_c)]
    if pick_p:
        view = view[view["pays"].isin(pick_p)]

    t1, t3, t4, t5 = st.tabs(["Tableau et carte", "Répartitions", "Rapport par entreprise", "Qualité"])
    with t1:
        cols = [c for c in DISPLAY_COLUMNS if c in view.columns]
        _stretch(st.dataframe, view[cols], hide_index=True,
                 column_config={"osm_url": st.column_config.LinkColumn("OSM", display_text="ouvrir")})
        d1, d2 = st.columns(2)
        d1.download_button("Télécharger (CSV)", _csv_bytes(view), "catmap_sites.csv", "text/csv")
        d2.download_button("Télécharger (Excel)", _xlsx_bytes({"sites": view, "rapport": report}), "catmap_sites.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.subheader("Carte interactive")
        _site_map(view)
    with t3:
        a, b = st.columns(2)
        with a:
            one = st.selectbox("Entreprise (répartition par pays)", ["Toutes"] + companies, key="pie_company")
            _pie(view if one == "Toutes" else view[view["nom_demande"] == one], "pays", "Répartition par pays")
        with b:
            cs = ["Tous"] + sorted(view["pays"].dropna().unique())
            one_c = st.selectbox("Pays (répartition par secteur NAF)", cs, key="pie_country")
            _pie(view if one_c == "Tous" else view[view["pays"] == one_c], "naf_section", "Répartition par section NAF")
    with t4:
        _render_report_tab(res)
    with t5:
        _render_quality_tab(view)


def _render_report_tab(res):
    st.subheader("Synthèse par entreprise")
    _stretch(st.dataframe, summary_by_company(res.sites, res.report), hide_index=True)
    st.subheader("Détail de la recherche")
    _stretch(st.dataframe, res.report, hide_index=True)
    st.caption(f"Serveur(s) : {', '.join(res.meta['endpoints']) or 'cache'} — options : {res.meta['options']}")


def _render_quality_tab(view: pd.DataFrame):
    st.markdown("**Fiabilité de l'appariement du nom** (plus le score est élevé, plus le site est sûr)")
    bins = pd.cut(view["match_score"], [0, .5, .7, .8, .95, 1.0], labels=["≤0,5", "0,5-0,7", "0,7-0,8", "0,8-0,95", "1,0"])
    _stretch(st.dataframe, bins.value_counts().sort_index().rename("sites").reset_index().rename(columns={"match_score": "score"}), hide_index=True)
    st.markdown("**Fiabilité de la classification NAF**")
    _stretch(st.dataframe, view["naf_confidence"].fillna("non classé").value_counts().rename("sites").reset_index(), hide_index=True)
    un = get_naf().unmapped(view)
    if not un.empty:
        st.markdown("**Valeurs de tags OSM absentes de `data/naf_mapping.csv`** (à compléter)")
        _stretch(st.dataframe, un.head(50), hide_index=True)
    snapped = int((view["pays_methode"] == "nearest").sum())
    if snapped:
        st.caption(f"{snapped} site(s) rattaché(s) au pays le plus proche (point côtier ou frontalier).")


def _clear_all():
    """Vide le cache disque Overpass, les résultats affichés, les filtres et la saisie (appelé par le bouton)."""
    n = 0
    cache_dir = Path(get_settings().cache_dir)
    if cache_dir.is_dir():
        for f in list(cache_dir.glob("*.json")) + list(cache_dir.glob("*.tmp")):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    for k in ("result", "result_options", "result_queries", "f_company", "f_country", "pie_company", "pie_country",
              "map_color", "map_cluster"):
        st.session_state.pop(k, None)
    for k in ("in_name", "in_qid", "in_text"):
        if k in st.session_state:
            st.session_state[k] = ""
    try:
        st.cache_data.clear()
    except Exception:  # noqa: BLE001
        pass
    st.session_state["_cleared"] = n


def page_geolocalisation():
    st.title("Géolocalisation d'entreprises")
    mode = st.radio("Mode de saisie", ["Un nom d'entreprise", "Liste (une entreprise par ligne)", "Fichier CSV / Excel"], horizontal=True)
    queries, warns = _collect_queries(mode)
    for w in warns[:5]:
        st.warning(w)
    options = _options_ui()
    with st.expander("Tester la connexion Overpass"):
        if st.button("Lancer le test"):
            with st.spinner("Test des serveurs…"):
                rows = get_client().ping()
            for r in rows:
                (st.success if r["ok"] else st.error)(f"{r['endpoint']} — {r['latence_s']} s — {r['message']}")
            if not any(r["ok"] for r in rows):
                st.info("Aucun serveur joignable depuis cet hébergement. Voir le README : instance Overpass "
                        "payante/auto-hébergée accessible depuis internet, variable `OVERPASS_URLS`.")
    b1, b2 = st.columns([1, 1])
    run = b1.button("Lancer la recherche", type="primary", disabled=not queries)
    b2.button("Vider le cache et les résultats", on_click=_clear_all,
              help="Supprime les réponses Overpass mémorisées, les résultats affichés et la saisie. "
                   "La prochaine recherche interrogera de nouveau Overpass.")
    if "_cleared" in st.session_state:
        st.success(f"Cache vidé ({st.session_state.pop('_cleared')} fichier(s) supprimé(s)). Saisie et résultats réinitialisés.")
    if run:
        _run_search(queries, options)
    res = st.session_state.get("result")
    if res is not None:
        st.divider()
        shown = sorted(res.report["nom_demande"].astype(str).unique()) if "nom_demande" in res.report else []
        if shown:
            st.markdown("**Résultats pour :** " + ", ".join(shown[:10]) + (" …" if len(shown) > 10 else ""))
        current = [(q.raw.casefold(), q.wikidata) for q in queries]
        if current and current != st.session_state.get("result_queries"):
            st.warning("Ces résultats correspondent à la recherche précédente, pas à la saisie actuelle. "
                       "Cliquez sur « Lancer la recherche » pour les mettre à jour.")
        _render_result(res)


# --------------------------------------------------------------------------
# Pages « Temps réel » et « Cartographies avancées »
# --------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Interrogation de Météo-France…")
def _fetch_vigilance_df() -> pd.DataFrame:
    s = get_settings()
    return vg.parse_vigilance(vg.fetch_vigilance(vg.MeteoFranceAuth(s), s))


def _load_vigilance():
    s = get_settings()
    if not vg.MeteoFranceAuth(s).configured:
        st.info("Identifiants Météo-France absents. Définissez `METEOFRANCE_API_KEY` (ou `METEOFRANCE_CLIENT_ID` et "
                "`METEOFRANCE_CLIENT_SECRET`) dans l'environnement ou `.streamlit/secrets.toml`.")
        return None
    if st.button("Actualiser"):
        _fetch_vigilance_df.clear()
    try:
        return _fetch_vigilance_df()
    except vg.VigilanceError as e:
        st.error(str(e))
        return None


def page_temps_reel():
    st.title("Vigilance Météo-France")
    df = _load_vigilance()
    if df is None:
        return
    periodes = sorted(p for p in df["periode"].unique() if p) or [""]
    per = st.selectbox("Échéance", periodes) if len(periodes) > 1 else periodes[0]
    worst = vg.worst_by_domain(df, per or None)
    n = worst["couleur_id"].value_counts()
    c = st.columns(4)
    for i, (cid, name) in enumerate(vg.COLORS.items()):
        c[i].metric(name, int(n.get(cid, 0)))
    try:
        deps = get_departements()
    except vg.VigilanceError as e:
        st.warning(f"Carte indisponible : {e}")
        deps = None
    if deps is not None:
        import folium
        from streamlit_folium import st_folium
        by_code = {}
        for r in worst.itertuples():
            by_code[str(r.domaine).strip().upper()] = r
        def color_for(code: str):
            for cand in vg._domain_candidates(code):
                if cand in by_code:
                    return vg.COLOR_HEX.get(int(by_code[cand].couleur_id), "#999999")
            return "#cccccc"
        m = folium.Map(location=[46.6, 2.5], zoom_start=6, tiles="CartoDB positron")
        folium.GeoJson(deps.geojson, style_function=lambda f: {
            "fillColor": color_for(str((f.get("properties") or {}).get("code", ""))),
            "color": "#333", "weight": 0.6, "fillOpacity": 0.65},
            tooltip=folium.GeoJsonTooltip(fields=[k for k in ("nom", "code") if k in (deps.geojson["features"][0].get("properties") or {})])
        ).add_to(m)
        st_folium(m, height=560, returned_objects=[], key="vigilance_map")
    _stretch(st.dataframe, worst[worst["couleur_id"] >= 2].sort_values("couleur_id", ascending=False), hide_index=True)
    with st.expander("Données brutes analysées"):
        _stretch(st.dataframe, df, hide_index=True)


def page_cartographies_avancees():
    st.title("Sites exposés à la vigilance en cours")
    res = st.session_state.get("result")
    if res is None or res.sites.empty:
        st.info("Lancez d'abord une géolocalisation (page « Géolocalisation ») : les sites trouvés seront croisés avec la vigilance.")
        return
    df = _load_vigilance()
    if df is None:
        return
    try:
        exp = vg.expose_sites(res.sites, df, get_departements())
    except vg.VigilanceError as e:
        st.error(str(e))
        return
    fr = exp[exp["pays_iso2"] == "FR"]
    alert = fr[fr["vigilance_couleur_id"].fillna(0) >= 3]
    c = st.columns(3)
    c[0].metric("Sites en France", len(fr))
    c[1].metric("En vigilance orange ou rouge", len(alert))
    c[2].metric("Hors France (non traités)", len(exp) - len(fr))
    if alert.empty:
        st.success("Aucun site en vigilance orange ou rouge.")
    else:
        cols = ["nom_demande", "name", "departement", "vigilance_couleur", "vigilance_phenomenes", "naf_section", "surface_m2", "osm_url"]
        _stretch(st.dataframe, alert[cols].sort_values("vigilance_couleur"), hide_index=True)
        _site_map(alert)
        st.download_button("Télécharger (CSV)", _csv_bytes(alert[cols]), "catmap_sites_vigilance.csv", "text/csv")


# --------------------------------------------------------------------------
PAGES = {"Géolocalisation": page_geolocalisation, "Cartographies avancées": page_cartographies_avancees,
         "Temps réel": page_temps_reel}


def main():
    st.set_page_config(page_title="CatMap Risk", page_icon=str(ASSETS / "Logo_CatMapV2.png"), layout="wide")
    css = ASSETS / "styles.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    with st.sidebar:
        logo = ASSETS / "Logo_CatMapV2.png"
        if logo.exists():
            st.image(str(logo))
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        sm = ASSETS / "SquareManagement.png"
        if sm.exists():
            st.image(str(sm))
    PAGES[page]()


if __name__ == "__main__":
    main()
