"""Carte Leaflet autonome des cyclones actifs (HTML généré, affiché via components.html).

Pourquoi du Leaflet « à la main » plutôt que folium : sélecteur de couches directement sur la
carte, icônes SVG (symbole + numéro de catégorie) et popups entièrement maîtrisés, sans
dépendance supplémentaire. Toute la logique métier (icônes, popups, segments colorés) est
préparée côté Python et testable ; le JavaScript ne fait que dessiner.
"""
from __future__ import annotations

import html
import json
from typing import Optional

from .cyclones import (ICON_KEYS, Storm, TrackPoint, movement_text, pressure_text, stage_label,
                       wind_text)

ICON_COLOR = "#cd2a00"
# Silhouette du symbole fourni : voir _cyclone_icon.py (viewBox 20 105 570 400).
VIEWBOX = (20, 105, 570, 400)
HOLE_CENTER = (304.7, 304.5)
HOLE_RADIUS = 74.4

CAT_COLORS = {0: "#00c7d6", 1: "#ffe44d", 2: "#ffc233", 3: "#ff9a1f", 4: "#ff5a1f", 5: "#e0002b"}
CAT_LABELS = {0: "Tempête tropicale", 1: "Catégorie 1", 2: "Catégorie 2", 3: "Catégorie 3",
              4: "Catégorie 4", 5: "Catégorie 5"}

IMERG_TILES = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/IMERG_Precipitation_Rate/default/"
               "default/GoogleMapsCompatible_Level6/{z}/{y}/{x}.png")

LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"
LEAFLET_JS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"
CLUSTER_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.5.3/MarkerCluster.css"
CLUSTER_JS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.5.3/leaflet.markercluster.js"


def _load_path() -> str:
    from ._cyclone_icon import PATH
    return PATH


def icon_svg(key: str, width: int = 44) -> str:
    """Symbole ouragan (rouge) ; pour les catégories 1 à 5, le chiffre est placé au centre."""
    if key not in ICON_KEYS:
        raise ValueError(f"icône inconnue : {key!r}")
    x, y, w, h = VIEWBOX
    height = round(width * h / w)
    digit = ""
    if key != "TS":
        digit = (f'<text x="{HOLE_CENTER[0]}" y="{HOLE_CENTER[1] + 46}" text-anchor="middle" '
                 f'font-family="Arial,Helvetica,sans-serif" font-weight="800" font-size="130" '
                 f'fill="{ICON_COLOR}">{key}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="{x} {y} {w} {h}" role="img" aria-label="{html.escape(_alt(key))}">'
            f'<path fill="{ICON_COLOR}" fill-rule="evenodd" d="{_load_path()}"/>{digit}</svg>')


def _alt(key: str) -> str:
    return "Tempête tropicale" if key == "TS" else f"Ouragan, typhon ou cyclone de catégorie {key}"


def _row(label: str, value: str) -> str:
    return f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"


def popup_html(storm: Storm, p: TrackPoint, *, forecast: bool = False) -> str:
    stage = stage_label(p.wind_kt, storm.basin)
    when = p.valid.strftime("%d/%m %H:%M UTC") if p.valid else ""
    head = f"<div class='pn'>{html.escape(storm.name)}</div><div class='ps'>{html.escape(stage)}</div>"
    if forecast:
        head += f"<div class='pt'>Prévision +{p.tau} h{(' · ' + when) if when else ''}</div>"
    elif when:
        head += f"<div class='pt'>Position du {html.escape(when)}</div>"
    rows = [_row("Maximum Sustained Winds", wind_text(p.wind_kt)),
            _row("Max Wind Gusts", wind_text(p.gust_kt)),
            _row("Pressure", pressure_text(p.mslp_hpa)),
            _row("Movement", movement_text(p.direction_deg, p.speed_kt))]
    if p.precip_mm is not None:
        rows.append(_row("Expected rainfall (24 h)", f"{p.precip_mm:.0f} mm (au centre)"))
    foot = f"<div class='pf'>Bassin {html.escape(storm.basin or '?')}" + (
        f" · bulletin n°{html.escape(storm.advisory)}" if storm.advisory else "") + "</div>"
    return f"<div class='cy-pop'>{head}<table>{''.join(rows)}</table>{foot}</div>"


def _marker_html(name: str, key: str, size: int) -> tuple[str, int, int, int, int]:
    """HTML du marqueur : nom au-dessus, symbole dessous. Retourne (html, w, h, anchor_x, anchor_y)."""
    svg = icon_svg(key, size)
    svg_h = round(size * VIEWBOX[3] / VIEWBOX[2])
    label_h, w = 20, 150
    h = label_h + svg_h
    body = f"<div class='cy-name'>{html.escape(name)}</div><div class='cy-ico'>{svg}</div>"
    return body, w, h, w // 2, label_h + svg_h // 2


IMPACT_COLORS = ((24, "#d7191c"), (48, "#fd8d3c"), (72, "#fecc5c"), (10**6, "#ffffb2"))
NO_IMPACT_COLOR = "#2e9e5b"
NEUTRAL_COLOR = "#1f6fb5"
NAME_COLUMNS = ("name", "nom_demande", "nom", "entreprise", "company", "site", "label")


def impact_color(impact) -> str:
    try:
        i = int(impact)
    except (TypeError, ValueError):
        return NEUTRAL_COLOR
    if i <= 0:
        return NO_IMPACT_COLOR
    return next(c for lim, c in IMPACT_COLORS if i <= lim)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "None"
    return str(v)


def build_points(df, lat_col: str, lon_col: str, storms: list[Storm], *, analysed: bool = False,
                 cap: int = 5000) -> tuple[list[dict], bool]:
    """Points du CSV pour la carte. Avant analyse : bleus ; après : vert (hors trajectoire) ou rouge→jaune selon l'échéance.
    Retourne (points, tronqué). Au-delà de ``cap`` points, les points impactés sont gardés en priorité."""
    import pandas as pd
    lat = pd.to_numeric(df[lat_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    lon = pd.to_numeric(df[lon_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    ok = lat.between(-90, 90) & lon.between(-360, 360)
    idx = list(df.index[ok])
    if analysed and "impact" in df.columns:
        idx.sort(key=lambda i: 0 if (df.at[i, "impact"] or 0) > 0 else 1)
    truncated = len(idx) > cap
    idx = idx[:cap]
    track_lons = [p.lon for s in storms for p in s.forecast]
    east = any(x > 180 for x in track_lons)
    west = any(x < -180 for x in track_lons)
    name_col = next((c for c in df.columns if c.lower() in NAME_COLUMNS), None)
    if name_col is None:
        name_col = next((c for c in df.columns if c not in (lat_col, lon_col)), None)
    out = []
    for i in idx:
        x, y = float(lon[i]), float(lat[i])
        if east and x < -100:
            x += 360
        elif west and x > 100:
            x -= 360
        label = _fmt(df.at[i, name_col]) if name_col else f"Point {i + 1}"
        rows = [_row("Latitude / longitude", f"{y:.4f} / {float(lon[i]):.4f}")]
        color = NEUTRAL_COLOR
        if analysed and "impact" in df.columns:
            imp = int(df.at[i, "impact"] or 0)
            color = impact_color(imp)
            rows += [_row("Impact", f"{imp} h" if imp else "0 (hors trajectoire)"),
                     _row("Cyclone", _fmt(df.at[i, "cyclone"]) if "cyclone" in df.columns else "None"),
                     _row("max_sustained_winds", _fmt(df.at[i, "max_sustained_winds"] if "max_sustained_winds" in df.columns else None) + (" km/h" if imp else "")),
                     _row("max_wind_gusts", _fmt(df.at[i, "max_wind_gusts"] if "max_wind_gusts" in df.columns else None) + (" km/h" if imp else "")),
                     _row("total_precipitation", _fmt(df.at[i, "total_precipitation"] if "total_precipitation" in df.columns else None) + (" mm" if imp else ""))]
            tip = f"{label} — impact {imp} h" if imp else f"{label} — hors trajectoire"
        else:
            tip = label
        out.append({"lat": y, "lon": x, "color": color, "tip": tip,
                    "popup": f"<div class='cy-pop'><div class='pn'>{html.escape(label)}</div><table>{''.join(rows)}</table></div>"})
    return out, truncated


def build_payload(storms: list[Storm], radii: Optional[dict] = None, *, rain_available: bool = False,
                  points: Optional[list[dict]] = None, points_analysed: bool = False) -> dict:
    out_storms, segments, fpoints, past, rain = [], [], [], [], []
    for st in storms:
        if st.current is None:
            continue
        c = st.current
        body, w, h, ax, ay = _marker_html(st.name, c.icon_key, 46)
        out_storms.append({"name": st.name, "lat": c.lat, "lon": c.lon, "icon": c.icon_key, "html": body,
                           "w": w, "h": h, "ax": ax, "ay": ay, "popup": popup_html(st, c)})
        if len(st.past) > 1:
            past.append({"coords": [[a, b] for a, b in st.past], "name": st.name})
        fc = st.forecast
        for a, b in zip(fc, fc[1:]):
            segments.append({"coords": [[a.lat, a.lon], [b.lat, b.lon]], "color": CAT_COLORS[b.category],
                             "tip": f"{st.name} — +{b.tau} h : {wind_text(b.wind_kt)}"})
        for p in fc[1:]:
            svg = icon_svg(p.icon_key, 30)
            sh = round(30 * VIEWBOX[3] / VIEWBOX[2])
            tag = f"+{p.tau} h"
            fpoints.append({"lat": p.lat, "lon": p.lon, "icon": p.icon_key,
                            "html": f"<div class='cy-ico'>{svg}</div><div class='cy-tau'>{tag}</div>",
                            "w": 44, "h": sh + 14, "ax": 22, "ay": sh // 2, "popup": popup_html(st, p, forecast=True)})
        for p in fc:
            if p.precip_mm is not None:
                rain.append({"lat": p.lat, "lon": p.lon, "mm": p.precip_mm,
                             "tip": f"{st.name} — +{p.tau} h : {p.precip_mm:.0f} mm / 24 h (au centre)"})
    return {"storms": out_storms, "segments": segments, "forecast_points": fpoints, "past": past,
            "rain": rain, "radii": radii or {}, "rain_tiles": IMERG_TILES, "rain_available": rain_available,
            "colors": {str(k): v for k, v in CAT_COLORS.items()},
            "points": points or [], "points_analysed": points_analysed,
            "impact_colors": {"none": NO_IMPACT_COLOR, "le24": "#d7191c", "le48": "#fd8d3c", "le72": "#fecc5c", "later": "#ffffb2"}}


_CSS = """
html,body{margin:0;height:100%;font-family:Arial,Helvetica,sans-serif}
#map{position:absolute;inset:0;background:#dfe6ea}
.cy-marker{background:none;border:none}
.cy-name{height:20px;line-height:20px;text-align:center;font-weight:700;font-size:13px;color:#232323;
  text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff,0 0 4px #fff;white-space:nowrap}
.cy-ico{text-align:center;line-height:0;filter:drop-shadow(0 0 2px #fff) drop-shadow(0 0 1px #fff)}
.cy-tau{text-align:center;font-size:10px;font-weight:700;color:#232323;line-height:14px;
  text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff}
.cy-pop{min-width:250px;color:#232323}
.cy-pop .pn{font-size:16px;font-weight:700;color:#bd8e43}
.cy-pop .ps{font-weight:700;margin-bottom:2px}
.cy-pop .pt{color:#555;font-size:12px;margin-bottom:6px}
.cy-pop table{border-collapse:collapse;width:100%;margin-top:4px}
.cy-pop th{text-align:left;font-weight:700;padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap}
.cy-pop td{padding:3px 0}
.cy-pop tr+tr th,.cy-pop tr+tr td{border-top:1px solid #e5e5e5}
.cy-pop .pf{margin-top:6px;color:#777;font-size:11px}
.cy-legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.35);font-size:12px;line-height:18px;color:#232323}
.cy-legend b{display:block;margin-bottom:2px}
.cy-legend i{display:inline-block;width:22px;height:5px;margin-right:6px;vertical-align:middle;border:1px solid #333}
.cy-empty{position:absolute;z-index:1000;left:50%;top:14px;transform:translateX(-50%);background:#fff;padding:8px 14px;
  border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,.35);font-size:14px;color:#232323}
.leaflet-control-layers{color:#232323}
.cy-row{display:flex;align-items:center;justify-content:space-between;gap:14px}
.cy-ungroup{font-size:12px;color:#555;white-space:nowrap;cursor:pointer}
.cy-ungroup input{vertical-align:middle;margin:0 2px 0 0}
.cy-cluster div{border-radius:50%;border:2px solid #222;text-align:center;font-weight:700;font-size:12px;color:#222;box-shadow:0 0 4px rgba(0,0,0,.5)}
"""

_JS = r"""
const D = __PAYLOAD__;
const map = L.map('map', {worldCopyJump: true, minZoom: 2, zoomControl: true}).setView([18, -40], 3);
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
const base = {
  'Fond clair': L.tileLayer(ESRI + 'Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      {attribution: 'Tiles &copy; Esri', maxNativeZoom: 16, maxZoom: 18}),
  'Océans': L.tileLayer(ESRI + 'Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
      {attribution: 'Tiles &copy; Esri', maxNativeZoom: 13, maxZoom: 18}),
  'Satellite': L.tileLayer(ESRI + 'World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {attribution: 'Tiles &copy; Esri', maxZoom: 18}),
  'OpenStreetMap': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      {attribution: '&copy; OpenStreetMap', maxZoom: 18}),
  'Fond sombre': L.tileLayer(ESRI + 'Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      {attribution: 'Tiles &copy; Esri', maxNativeZoom: 16, maxZoom: 18})
};
base['Fond clair'].addTo(map);

function mk(o) {
  return L.marker([o.lat, o.lon], {
    icon: L.divIcon({className: 'cy-marker', html: o.html, iconSize: [o.w, o.h], iconAnchor: [o.ax, o.ay],
                     popupAnchor: [0, -(o.ay) + 4]}),
    riseOnHover: true
  }).bindPopup(o.popup, {maxWidth: 340});
}

const gStorms = L.layerGroup(), gTrack = L.layerGroup(), gPast = L.layerGroup(),
      gRadii = L.layerGroup(), gRain = L.layerGroup(), gImerg = L.layerGroup();

D.storms.forEach(s => mk(s).addTo(gStorms));
D.segments.forEach(s => {
  L.polyline(s.coords, {color: '#222', weight: 7, opacity: .55}).addTo(gTrack);
  L.polyline(s.coords, {color: s.color, weight: 4, opacity: 1, dashArray: '8 6'}).bindTooltip(s.tip, {sticky: true}).addTo(gTrack);
});
D.forecast_points.forEach(p => mk(p).addTo(gTrack));
D.past.forEach(p => L.polyline(p.coords, {color: '#555', weight: 3, opacity: .9}).bindTooltip('Trace passée — ' + p.name, {sticky: true}).addTo(gPast));

const RC = {34: '#ffd400', 50: '#ff8a00', 64: '#e0002b'};
if (D.radii && D.radii.features) {
  L.geoJSON(D.radii, {style: f => ({color: RC[f.properties.kt] || '#999', weight: 1.5, fillColor: RC[f.properties.kt] || '#999', fillOpacity: .18}),
    onEachFeature: (f, l) => l.bindTooltip('Vent ≥ ' + f.properties.kt + ' kt (rayon actuel)', {sticky: true})}).addTo(gRadii);
}
function rainColor(mm) { return mm >= 200 ? '#08306b' : mm >= 100 ? '#2171b5' : mm >= 50 ? '#4292c6' : mm >= 20 ? '#6baed6' : '#c6dbef'; }
D.rain.forEach(r => L.circleMarker([r.lat, r.lon], {radius: 7 + Math.min(18, Math.sqrt(r.mm)), color: '#08306b', weight: 1,
  fillColor: rainColor(r.mm), fillOpacity: .6}).bindTooltip(r.tip, {sticky: true}).addTo(gRain));
L.tileLayer(D.rain_tiles, {opacity: .7, maxNativeZoom: 6, maxZoom: 18, attribution: 'NASA GIBS / IMERG'}).addTo(gImerg);

const gPts = L.layerGroup();
const RANK = {'#d7191c': 5, '#fd8d3c': 4, '#fecc5c': 3, '#ffffb2': 2, '#2e9e5b': 1, '#1f6fb5': 0};
const ptMarkers = (D.points || []).map(p => {
  const mk = L.circleMarker([p.lat, p.lon], {radius: 6, color: '#222', weight: 1.2, fillColor: p.color, fillOpacity: .95});
  mk.options.rank = RANK[p.color] || 0; mk.options.col = p.color;
  return mk.bindPopup(p.popup, {maxWidth: 340}).bindTooltip(p.tip, {sticky: true});
});
const canCluster = typeof L.markerClusterGroup === 'function';
const cluster = canCluster ? L.markerClusterGroup({
  maxClusterRadius: 45, showCoverageOnHover: false,
  iconCreateFunction: c => {
    const ms = c.getAllChildMarkers();
    const worst = ms.reduce((a, b) => (b.options.rank > a.options.rank ? b : a), ms[0]);
    const n = c.getChildCount();
    const s = n < 10 ? 30 : n < 100 ? 36 : 42;
    return L.divIcon({className: 'cy-cluster', iconSize: [s, s],
      html: '<div style="width:' + s + 'px;height:' + s + 'px;line-height:' + (s - 4) + 'px;background:' + worst.options.col + '">' + n + '</div>'});
  }}) : null;
const flat = L.layerGroup();
let grouped = canCluster;
function renderPoints() {
  gPts.clearLayers();
  if (grouped) { cluster.clearLayers(); cluster.addLayers(ptMarkers); gPts.addLayer(cluster); }
  else { flat.clearLayers(); ptMarkers.forEach(m => flat.addLayer(m)); gPts.addLayer(flat); }
}
renderPoints();
const ungroupBoxes = [];
function setGrouped(g) { grouped = g; ungroupBoxes.forEach(b => { b.checked = !g; }); renderPoints(); }
function ungroupToggle(extraClass) {
  const l = document.createElement('label');
  l.className = 'cy-ungroup ' + (extraClass || '');
  const b = document.createElement('input'); b.type = 'checkbox'; b.checked = !grouped;
  b.addEventListener('change', () => setGrouped(!b.checked));
  ungroupBoxes.push(b);
  l.appendChild(b); l.appendChild(document.createTextNode(' Dégrouper'));
  ['click', 'dblclick', 'mousedown'].forEach(ev => l.addEventListener(ev, e => e.stopPropagation()));
  return l;
}

const overlays = {
  'Cyclones actifs (position, nom)': gStorms,
  'Trajectoire prévue et vents': gTrack,
  'Trace passée': gPast,
  'Zones de vent actuelles (34 / 50 / 64 kt)': gRadii,
  'Précipitations attendues (24 h, modèle)': gRain,
  'Précipitations observées (NASA IMERG)': gImerg
};
if (D.points.length) overlays['Vos points'] = gPts;
gStorms.addTo(map); gTrack.addTo(map); gPast.addTo(map); gRadii.addTo(map);
if (D.rain.length) gRain.addTo(map);
if (D.points.length) gPts.addTo(map);
const layersCtl = L.control.layers(base, overlays, {collapsed: false, position: 'topright'}).addTo(map);
function decorateCtl() {
  if (!canCluster || !D.points.length) return;
  const el = layersCtl.getContainer();
  el.querySelectorAll('.leaflet-control-layers-overlays label').forEach(lab => {
    if (lab.textContent.trim() !== 'Vos points' || lab.parentNode.classList.contains('cy-row')) return;
    const row = document.createElement('div'); row.className = 'cy-row';
    lab.parentNode.insertBefore(row, lab); row.appendChild(lab); row.appendChild(ungroupToggle('in-ctl'));
  });
}
if (layersCtl._update) {            // Leaflet reconstruit la liste à chaque ajout/retrait de couche : on re-décore
  const upd = layersCtl._update;
  layersCtl._update = function () { const r = upd.apply(this, arguments); for (let i = ungroupBoxes.length - 1; i >= 0; i--) if (!ungroupBoxes[i].isConnected) ungroupBoxes.splice(i, 1); decorateCtl(); return r; };
}
decorateCtl();

const legend = L.control({position: 'bottomleft'});
legend.onAdd = function () {
  const d = L.DomUtil.create('div', 'cy-legend');
  const names = {0: 'Tempête tropicale', 1: 'Catégorie 1', 2: 'Catégorie 2', 3: 'Catégorie 3', 4: 'Catégorie 4', 5: 'Catégorie 5'};
  d.innerHTML = '<b>Vent prévu le long de la trajectoire</b>' +
    Object.keys(names).map(k => '<div><i style="background:' + D.colors[k] + '"></i>' + names[k] + '</div>').join('');
  L.DomEvent.disableClickPropagation(d);
  if (D.points.length) {
    const c = D.impact_colors, dot = col => '<i style="background:' + col + ';width:12px;height:12px;border-radius:50%"></i>';
    d.innerHTML += D.points_analysed
      ? '<b style="margin-top:6px">Vos points</b><div>' + dot(c.none) + 'Hors trajectoire</div><div>' + dot(c.le24) + 'Impact ≤ 24 h</div><div>' + dot(c.le48) + 'Impact ≤ 48 h</div><div>' + dot(c.le72) + 'Impact ≤ 72 h</div><div>' + dot(c.later) + 'Impact &gt; 72 h</div>'
      : '<b style="margin-top:6px">Vos points</b><div>' + dot('#1f6fb5') + 'Non analysés</div>';
    if (canCluster) { const t = ungroupToggle('in-legend'); t.style.display = 'block'; t.style.marginTop = '4px'; d.appendChild(t); }
  }
  return d;
};
legend.addTo(map);

const all = [].concat(D.storms.map(s => [s.lat, s.lon]), D.segments.flatMap(s => s.coords), D.points.map(p => [p.lat, p.lon]));
if (all.length) { map.fitBounds(L.latLngBounds(all).pad(0.2), {maxZoom: 6}); }
else {
  const e = document.createElement('div'); e.className = 'cy-empty';
  e.textContent = 'Aucun cyclone tropical actif signalé actuellement.'; document.body.appendChild(e);
}
"""


def render_map_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    js = _JS.replace("__PAYLOAD__", data)
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<link rel='stylesheet' href='{LEAFLET_CSS}'><link rel='stylesheet' href='{CLUSTER_CSS}'><style>{_CSS}</style></head>"
            f"<body><div id='map'></div><script src='{LEAFLET_JS}'></script><script src='{CLUSTER_JS}'></script><script>{js}</script></body></html>")
