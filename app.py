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


def _site_map(df: pd.DataFrame, color_col: str = "nom_demande"):
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium

    d = df.dropna(subset=["lat", "long"])
    if len(d) > MAX_MARKERS:
        st.warning(f"Carte limitée à {MAX_MARKERS} sites sur {len(d)} (les mieux appariés). "
                   "Filtrez par entreprise ou par pays pour tout voir.")
        d = d.sort_values("match_score", ascending=False).head(MAX_MARKERS)
    if d.empty:
        st.info("Aucun site à afficher.")
        return
    m = folium.Map(location=[float(d["lat"].mean()), float(d["long"].mean())], zoom_start=5, control_scale=True)
    m.fit_bounds([[float(d["lat"].min()), float(d["long"].min())], [float(d["lat"].max()), float(d["long"].max())]])
    cluster = MarkerCluster().add_to(m)
    colors = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(sorted(d[color_col].dropna().unique()))}
    for _, r in d.iterrows():
        folium.CircleMarker(
            [float(r["lat"]), float(r["long"])], radius=6, color=colors.get(r[color_col], "#1f77b4"),
            fill=True, fill_opacity=0.7, popup=folium.Popup(_popup_html(r), max_width=320),
            tooltip=html.escape(str(r["name"])),
        ).add_to(cluster)
    st_folium(m, height=620, returned_objects=[], key=f"map_{color_col}_{len(d)}")


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
    if s.uses_public_overpass:
        st.caption("Serveurs Overpass **publics** : usage raisonnable demandé. "
                   + ("" if s.contact else "Renseignez `CATMAP_CONTACT` (e-mail ou URL) pour vous identifier. ")
                   + "Pour de gros volumes, voir « Overpass local » dans le README.")
    else:
        st.caption(f"Instance Overpass **locale** : {s.overpass_urls[0]}")
    return SearchOptions(include_operator=op, include_relations=rel, require_business_tag=biz,
                         countries=tuple(p.rsplit("(", 1)[1].rstrip(")") for p in picked),
                         strict_word_match=strict, min_score=min_score, dedup_radius_m=float(dedup))


def _run_search(queries: list[NameQuery], options: SearchOptions):
    box = st.empty()
    bar = box.progress(0.0, text="Démarrage…")

    def cb(done: int, total: int, name: str):
        bar.progress(done / total, text=f"{done}/{total} — {name}")

    res = geolocate(queries, options, get_client(), get_settings(), get_countries(), get_naf(), progress=cb)
    box.empty()
    st.session_state["result"] = res
    st.session_state["result_options"] = options


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

    t1, t2, t3, t4, t5 = st.tabs(["Tableau", "Carte", "Répartitions", "Rapport par entreprise", "Qualité"])
    with t1:
        cols = [c for c in DISPLAY_COLUMNS if c in view.columns]
        _stretch(st.dataframe, view[cols], hide_index=True,
                 column_config={"osm_url": st.column_config.LinkColumn("OSM", display_text="ouvrir")})
        d1, d2 = st.columns(2)
        d1.download_button("Télécharger (CSV)", _csv_bytes(view), "catmap_sites.csv", "text/csv")
        d2.download_button("Télécharger (Excel)", _xlsx_bytes({"sites": view, "rapport": report}), "catmap_sites.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with t2:
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


def page_geolocalisation():
    st.title("Géolocalisation d'entreprises")
    mode = st.radio("Mode de saisie", ["Un nom d'entreprise", "Liste (une entreprise par ligne)", "Fichier CSV / Excel"], horizontal=True)
    queries, warns = _collect_queries(mode)
    for w in warns[:5]:
        st.warning(w)
    options = _options_ui()
    if st.button("Lancer la recherche", type="primary", disabled=not queries):
        _run_search(queries, options)
    res = st.session_state.get("result")
    if res is not None:
        st.divider()
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
