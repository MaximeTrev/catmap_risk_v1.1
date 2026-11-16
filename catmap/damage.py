"""Dommages d'inondation à partir des hauteurs d'eau : courbes hauteur → taux de dommage.

Deux familles de courbes au choix :

* **JRC** (Huizinga, De Moel, Szewczyk, 2017, EUR 28552 EN, « Global flood depth-damage functions ») : une courbe par continent et par
  classe d'actif (résidentiel, commercial, industriel, transport, infrastructure, agriculture), hauteur en mètres, taux de dommage
  entre 0 et 1 ; valeurs de remplacement maximales par pays (€/m²).
* **FEMA Hazus** (fonctions riverines, reprises sur Zenodo par A. Pollack, 2023, DOI 10.5281/zenodo.10027236, CC BY 4.0) : fichier
  ``haz_fl_dept.csv`` en format large (colonnes ``ft04m`` … ``ft24`` = pieds par rapport au plancher du rez-de-chaussée, valeurs en %).

Aucun jeu de courbes n'est « inventé » ici : les fichiers sont lus tels quels (dossier ``data/damage/``, téléchargement Zenodo pour
Hazus, ou téléversement). Le format du fichier Excel JRC n'a PAS pu être vérifié depuis l'environnement de développement (réseau
bloqué) : ``load_jrc`` est tolérant et renvoie un diagnostic ; à défaut, un CSV « long » simple est accepté (voir ``JRC_TEMPLATE``).

Perte = taux de dommage(hauteur) × valeur exposée. Perte annuelle moyenne (PAM) : intégrale trapézoïdale de la perte sur la
probabilité annuelle 1/RP, entre les périodes de retour calculées (sans extrapolation hors de cet intervalle).
"""
from __future__ import annotations

import io
import math
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

FT = 0.3048
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "damage"
CACHE_DIR = Path(tempfile.gettempdir()) / "catmap_damage"
HAZUS_URL = "https://zenodo.org/records/10027236/files/{name}?download=1"
HAZUS_FILES = ("haz_fl_dept.csv", "haz_fl_occ.csv", "riverine_ids.csv")

CLASSES = ("residential", "commercial", "industrial", "transport", "infrastructure", "agriculture")
CLASS_LABELS = {"residential": "Résidentiel", "commercial": "Commercial", "industrial": "Industriel", "transport": "Transport",
                "infrastructure": "Infrastructure", "agriculture": "Agriculture"}
_CLASS_KEYS = {"residential": ("resid", "habit", "logement"), "commercial": ("commerc", "retail", "shop", "bureau", "office"),
               "industrial": ("industr", "usine", "factory"), "transport": ("transport",), "infrastructure": ("infrastr",),
               "agriculture": ("agricul",)}
CONTINENTS = {"europe": "Europe", "asia": "Asia", "africa": "Africa", "oceania": "Oceania", "australia": "Oceania",
              "north america": "North America", "south america": "South America", "latin america": "South America"}

JRC_TEMPLATE = ("Courbes JRC au format CSV « long » accepté : colonnes `continent`, `classe` (residential / commercial / industrial / "
                "transport / infrastructure / agriculture), `hauteur_m`, `taux` (0 à 1) ; valeurs maximales : colonnes `pays`, `classe`, "
                "`eur_m2` (fichier séparé, optionnel).")


class DamageError(Exception):
    """Fichier de courbes absent, illisible ou inexploitable."""


def class_of(text) -> Optional[str]:
    """Classe d'actif d'après un libellé libre (« Industrial buildings », « usine », « COM1 »…)."""
    t = str(text or "").strip().lower()
    if not t or t == "nan":
        return None
    for cls, keys in _CLASS_KEYS.items():
        if any(k in t for k in keys):
            return cls
    if re.match(r"^res\d", t):
        return "residential"
    if re.match(r"^com\d", t):
        return "commercial"
    if re.match(r"^ind\d", t):
        return "industrial"
    return None


# ───────────────────────────── courbes (structure commune) ─────────────────────────────
class Curve:
    """Courbe hauteur (m, relative à la référence de la source) → taux de dommage (0 à 1)."""

    def __init__(self, depth_m, ratio, label: str = "", n_curves: int = 1):
        d = np.asarray(depth_m, dtype=float)
        r = np.asarray(ratio, dtype=float)
        ok = np.isfinite(d) & np.isfinite(r)
        order = np.argsort(d[ok])
        self.depth = d[ok][order]
        self.ratio = np.clip(r[ok][order], 0.0, 1.0)
        self.label = label
        self.n_curves = n_curves
        if len(self.depth) < 2:
            raise DamageError(f"Courbe inexploitable ({label or 'sans nom'}) : moins de 2 points.")

    def at(self, depth_m) -> np.ndarray:
        """Taux de dommage ; en dessous du 1er point : 0 ; au-delà du dernier : valeur du dernier point (plafond)."""
        x = np.asarray(depth_m, dtype=float)
        out = np.interp(x, self.depth, self.ratio, left=0.0, right=float(self.ratio[-1]))
        return np.where(np.isfinite(x), out, np.nan)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"hauteur (m)": self.depth, "taux de dommage": self.ratio})


# ───────────────────────────── Hazus ─────────────────────────────
_HAZ_COL = re.compile(r"^ft(\d+)(m)?(?:_(\d+))?$", re.I)


def _hazus_depth_ft(col: str) -> Optional[float]:
    m = _HAZ_COL.match(str(col).strip())
    if not m:
        return None
    v = float(m.group(1)) + (0.5 if m.group(3) == "5" else 0.0)
    return -v if m.group(2) else v


def parse_hazus(dept_bytes: bytes, riverine_ids_bytes: Optional[bytes] = None, occ_bytes: Optional[bytes] = None) -> dict:
    """Lit ``haz_fl_dept.csv`` → ``{"curves": DataFrame long, "occupancies": {code: libellé}, "ids": set|None}``.

    Le DataFrame long contient ``occupancy, dmgfn, source, description, depth_ft, pct`` pour les fonctions « structure » du bâti."""
    try:
        raw = pd.read_csv(io.BytesIO(dept_bytes), encoding="utf-8-sig", low_memory=False)
    except Exception as e:                                   # noqa: BLE001
        raise DamageError(f"haz_fl_dept.csv illisible : {e}") from e
    depth_cols = {c: _hazus_depth_ft(c) for c in raw.columns}
    depth_cols = {c: d for c, d in depth_cols.items() if d is not None}
    if "Occupancy" not in raw.columns or len(depth_cols) < 5:
        raise DamageError("haz_fl_dept.csv : colonnes « Occupancy » et « ftNN » introuvables (fichier différent de celui attendu ?).")
    if "Source_Table" in raw.columns:
        raw = raw[raw["Source_Table"].astype(str).str.contains("BldgStruct", case=False, na=False)]
    elif "Cover_Class" in raw.columns:
        raw = raw[raw["Cover_Class"].astype(str).str.lower().eq("bldg")]
    long = raw.melt(id_vars=[c for c in ("Occupancy", "DmgFnId", "Source", "Description") if c in raw.columns],
                    value_vars=list(depth_cols), var_name="col", value_name="pct")
    long["depth_ft"] = long["col"].map(depth_cols)
    long["pct"] = pd.to_numeric(long["pct"], errors="coerce")
    long = long.dropna(subset=["pct"]).drop(columns="col")
    long = long.rename(columns={"Occupancy": "occupancy", "DmgFnId": "dmgfn", "Source": "source", "Description": "description"})
    for c in ("dmgfn", "source", "description"):
        if c not in long.columns:
            long[c] = ""
    occs = {str(o): "" for o in long["occupancy"].dropna().unique()}
    if occ_bytes:
        try:
            o = pd.read_csv(io.BytesIO(occ_bytes), encoding="utf-8-sig")
            code_c = next((c for c in o.columns if c.lower() in ("occupancy", "occupy", "code", "occ")), o.columns[0])
            txt_c = next((c for c in o.columns if c != code_c and o[c].dtype == object), None)
            if txt_c:
                for code, txt in zip(o[code_c].astype(str), o[txt_c].astype(str)):
                    if code in occs:
                        occs[code] = txt
        except Exception:                                    # noqa: BLE001
            pass
    ids = None
    if riverine_ids_bytes:
        try:
            r = pd.read_csv(io.BytesIO(riverine_ids_bytes), encoding="utf-8-sig")
            col = next((c for c in r.columns if re.search(r"dmg\s*fn|damage.*id|^id$", str(c), re.I)), None)
            if col is not None:
                ids = set(pd.to_numeric(r[col], errors="coerce").dropna().astype(int))
        except Exception:                                    # noqa: BLE001
            ids = None
    return {"curves": long, "occupancies": dict(sorted(occs.items())), "ids": ids}


def hazus_curve(hz: dict, occupancy: str, floor_height_m: float = 0.0, only_riverine_defaults: bool = True) -> Curve:
    """Courbe médiane (par hauteur) des fonctions Hazus de l'occupation donnée, exprimée en hauteur d'eau **au-dessus du terrain**.

    Les fonctions Hazus sont relatives au plancher du rez-de-chaussée : hauteur terrain = hauteur Hazus + hauteur du plancher."""
    c = hz["curves"]
    c = c[c["occupancy"].astype(str) == str(occupancy)]
    used_defaults = False
    if only_riverine_defaults and hz.get("ids"):
        sub = c[pd.to_numeric(c["dmgfn"], errors="coerce").isin(hz["ids"])]
        if not sub.empty:
            c, used_defaults = sub, True
    if c.empty:
        raise DamageError(f"Hazus : aucune fonction pour l'occupation « {occupancy} ».")
    piv = c.pivot_table(index="depth_ft", columns="dmgfn", values="pct", aggfunc="first")
    piv = piv.sort_index().interpolate(method="index", limit_area="inside")
    med = piv.median(axis=1, skipna=True).dropna()
    n = piv.shape[1]
    lab = f"Hazus {occupancy} ({n} fonction(s)" + (", fonctions riverines par défaut" if used_defaults else "") + ")"
    return Curve(med.index.to_numpy() * FT + float(floor_height_m), med.to_numpy() / 100.0, lab, n)


# ───────────────────────────── JRC ─────────────────────────────
def _num(x) -> Optional[float]:
    try:
        v = float(str(x).replace(",", ".").replace(" ", "").strip())
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _continent_of(text) -> Optional[str]:
    t = str(text or "").strip().lower()
    for k, v in CONTINENTS.items():
        if t == k or (len(t) < 40 and k in t):
            return v
    return None


def _grid_to_jrc(grid: pd.DataFrame, sheet: str, curves: dict, maxdmg: dict, notes: list) -> None:
    g = grid.reset_index(drop=True)
    n_r, n_c = g.shape
    cont_here = _continent_of(sheet)
    is_max_sheet = "max" in sheet.lower()
    r = 0
    while r < n_r:
        row = [str(v) for v in g.iloc[r].tolist()]
        cls_cols = {j: class_of(v) for j, v in enumerate(row)}
        cls_cols = {j: c for j, c in cls_cols.items() if c}
        depth_col = next((j for j, v in enumerate(row) if "depth" in v.lower() or "hauteur" in v.lower()), None)
        for v in row:                                         # un continent seul sur une ligne = titre du bloc suivant
            cc = _continent_of(v)
            if cc and len(cls_cols) < 2:
                cont_here = cc
        if len(cls_cols) >= 2 and depth_col is not None and not is_max_sheet:
            cont = cont_here
            if cont is None:
                for k in range(max(0, r - 3), r + 1):         # titre éventuel juste au-dessus de l'en-tête
                    for v in g.iloc[k].tolist():
                        cont = cont or _continent_of(v)
            rr = r + 1
            rows = []
            while rr < n_r:
                d = _num(g.iat[rr, depth_col])
                if d is None:
                    break
                rows.append((d, {j: _num(g.iat[rr, j]) for j in cls_cols}))
                rr += 1
            if rows and cont:
                for j, cls in cls_cols.items():
                    pts = [(d, vals[j]) for d, vals in rows if vals[j] is not None]
                    if len(pts) >= 2:
                        top = max(p[1] for p in pts)
                        scale = 100.0 if top > 1.5 else 1.0
                        curves[(cont, cls)] = ([p[0] for p in pts], [p[1] / scale for p in pts])
            elif rows:
                notes.append(f"{sheet} : bloc de courbes sans continent identifiable (ignoré).")
            r = rr
            continue
        if is_max_sheet and len(cls_cols) >= 1:
            rr = r + 1
            while rr < n_r:
                country = str(g.iat[rr, 0]).strip()
                if not country or country.lower() == "nan":
                    break
                for j, cls in cls_cols.items():
                    v = _num(g.iat[rr, j])
                    if v is not None and v > 0:
                        maxdmg[(country.lower(), cls)] = v
                rr += 1
            r = rr
            continue
        r += 1


def parse_jrc(data: bytes, filename: str, maxdata: Optional[bytes] = None) -> dict:
    """Lit les courbes JRC (Excel officiel, tolérant ; ou CSV « long ») → ``{"curves": {(continent, classe): (h, taux)},
    "maxdmg": {(pays, classe): eur_m2}, "notes": [...]}``."""
    curves: dict = {}
    maxdmg: dict = {}
    notes: list = []
    name = filename.lower()
    try:
        if name.endswith((".xlsx", ".xlsm", ".xls")):
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
            for sh, grid in sheets.items():
                _grid_to_jrc(grid, str(sh), curves, maxdmg, notes)
        else:
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="utf-8-sig")
            df.columns = [str(c).strip().lower() for c in df.columns]
            need = {"continent", "classe", "hauteur_m", "taux"}
            if need <= set(df.columns):
                for (cont, cls), g in df.groupby(["continent", "classe"]):
                    cc, kk = _continent_of(cont), class_of(cls)
                    if cc and kk:
                        top = float(pd.to_numeric(g["taux"], errors="coerce").max())
                        curves[(cc, kk)] = (pd.to_numeric(g["hauteur_m"], errors="coerce").tolist(),
                                            (pd.to_numeric(g["taux"], errors="coerce") / (100.0 if top > 1.5 else 1.0)).tolist())
            elif {"pays", "classe", "eur_m2"} <= set(df.columns):
                for _, r in df.iterrows():
                    kk, v = class_of(r["classe"]), _num(r["eur_m2"])
                    if kk and v:
                        maxdmg[(str(r["pays"]).strip().lower(), kk)] = v
            else:
                notes.append("CSV JRC non reconnu. " + JRC_TEMPLATE)
    except DamageError:
        raise
    except Exception as e:                                   # noqa: BLE001
        raise DamageError(f"Fichier JRC illisible ({filename}) : {e}") from e
    if maxdata:
        try:
            m2 = parse_jrc(maxdata, "max.csv" if not str(filename).lower().endswith(("xlsx", "xls")) else filename)
            maxdmg.update(m2["maxdmg"])
        except DamageError as e:
            notes.append(str(e))
    if not curves and not maxdmg:
        notes.append("Aucune courbe reconnue dans le fichier. " + JRC_TEMPLATE)
    return {"curves": curves, "maxdmg": maxdmg, "notes": notes}


def jrc_curve(jrc: dict, continent: str, cls: str) -> Curve:
    key = (continent, cls)
    if key not in jrc["curves"]:
        have = sorted({f"{c} / {k}" for c, k in jrc["curves"]})
        raise DamageError(f"JRC : pas de courbe « {continent} / {cls} » dans le fichier (disponibles : {', '.join(have) or 'aucune'}).")
    h, r = jrc["curves"][key]
    return Curve(h, r, f"JRC {continent} – {CLASS_LABELS.get(cls, cls)}")


def jrc_max_value(jrc: dict, country: str, cls: str) -> Optional[float]:
    return jrc.get("maxdmg", {}).get((str(country or "").strip().lower(), cls))


# ───────────────────────────── chargement des fichiers ─────────────────────────────
def find_local(name: str) -> Optional[Path]:
    for d in (DATA_DIR, CACHE_DIR):
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def fetch_hazus(session=None, timeout: int = 30) -> dict:
    """Télécharge les fichiers Hazus depuis Zenodo (cache disque). Renvoie ``{nom: octets}`` (peut être partiel) ; lève
    ``DamageError`` si le fichier principal est introuvable."""
    import requests
    sess = session or requests.Session()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    errs = []
    for name in HAZUS_FILES:
        p = find_local(name)
        if p:
            out[name] = p.read_bytes()
            continue
        try:
            r = sess.get(HAZUS_URL.format(name=name), timeout=timeout)
            r.raise_for_status()
            if not r.content:
                raise DamageError("réponse vide")
            (CACHE_DIR / name).write_bytes(r.content)
            out[name] = r.content
        except Exception as e:                               # noqa: BLE001
            errs.append(f"{name} : {type(e).__name__} {str(e)[:120]}")
    if "haz_fl_dept.csv" not in out:
        raise DamageError("Téléchargement Hazus impossible (" + " ; ".join(errs) + "). Déposez haz_fl_dept.csv dans data/damage/ "
                          "(GitHub) ou téléversez-le ci-dessous.")
    return out


def local_files(names) -> dict:
    return {n: p.read_bytes() for n in names if (p := find_local(n))}


# ───────────────────────────── calcul ─────────────────────────────
def asset_class_series(df: pd.DataFrame, col: Optional[str], default: str) -> pd.Series:
    """Classe d'actif par point : colonne libre (libellés reconnus) sinon classe par défaut."""
    if not col or col not in df.columns:
        return pd.Series(default, index=df.index)
    return df[col].map(class_of).fillna(default)


def exposed_value(df: pd.DataFrame, mode: str, value_col: Optional[str] = None, area_col: Optional[str] = None,
                  unit_value: float = 0.0, const: float = 0.0, country_col: Optional[str] = None,
                  classes: Optional[pd.Series] = None, jrc: Optional[dict] = None) -> pd.Series:
    """Valeur exposée (€) par point. ``mode`` : ``col`` (colonne en €), ``area`` (surface × €/m² saisi), ``const`` (valeur unique),
    ``jrc_area`` (surface × valeur de remplacement JRC du pays)."""
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", ".").str.replace(r"[^\d.\-eE]", "", regex=True), errors="coerce")  # noqa: E731
    if mode == "col":
        return num(value_col)
    if mode == "area":
        return num(area_col) * float(unit_value)
    if mode == "const":
        return pd.Series(float(const), index=df.index)
    if mode == "jrc_area":
        if jrc is None or not country_col:
            raise DamageError("Valeurs JRC par pays : fichier des valeurs maximales et colonne pays requis.")
        vals = [jrc_max_value(jrc, c, k) for c, k in zip(df[country_col], classes)]
        return num(area_col) * pd.Series([v if v is not None else np.nan for v in vals], index=df.index)
    raise DamageError(f"Mode de valeur exposée inconnu : {mode}")


def column_ratio(rp: int) -> str:
    return f"dommage_RP{int(rp)}_ratio"


def column_eur(rp: int) -> str:
    return f"dommage_RP{int(rp)}_eur"


def estimate_damages(df: pd.DataFrame, depth_cols: dict, curves_by_class: dict, classes: pd.Series, values: pd.Series,
                     method_label: str) -> tuple[pd.DataFrame, dict]:
    """Calcule taux et pertes par scénario.

    ``depth_cols`` : ``{rp: nom de colonne de hauteur (m)}`` ; ``curves_by_class`` : ``{classe: Curve}``. Renvoie
    (DataFrame de nouvelles colonnes aligné sur ``df``, résumé)."""
    out = pd.DataFrame(index=df.index)
    out["dommage_methode"] = method_label
    out["dommage_classe"] = classes
    out["dommage_valeur_exposee_eur"] = values
    missing = sorted({c for c in classes.dropna().unique() if c not in curves_by_class})
    for rp, col in depth_cols.items():
        depth = pd.to_numeric(df[col], errors="coerce")
        ratio = pd.Series(np.nan, index=df.index)
        for cls, curve in curves_by_class.items():
            m = classes == cls
            if m.any():
                r = pd.Series(curve.at(depth[m].to_numpy()), index=depth[m].index)
                ratio.loc[m] = r.where(depth[m] > 0, 0.0).where(depth[m].notna())
        out[column_ratio(rp)] = ratio
        out[column_eur(rp)] = ratio * values
    rps = sorted(depth_cols)
    aal = None
    if len(rps) >= 2:
        p = np.array([1.0 / r for r in rps])[::-1]            # probabilités croissantes
        loss = np.column_stack([out[column_eur(r)].to_numpy(dtype=float) for r in rps][::-1])
        aal = np.trapezoid(loss, p, axis=1) if hasattr(np, "trapezoid") else np.trapz(loss, p, axis=1)
        out["dommage_PAM_eur"] = aal
    total = {column_eur(r): float(out[column_eur(r)].sum(skipna=True)) for r in rps}
    summ = {"points": len(df), "sans_valeur": int(values.isna().sum()), "sans_hauteur": int(
        pd.concat([pd.to_numeric(df[c], errors="coerce") for c in depth_cols.values()], axis=1).isna().all(axis=1).sum()),
            "classes_sans_courbe": missing, "pertes_totales": total, "pam_totale": None if aal is None else float(np.nansum(aal)),
            "pam_bornes": (rps[0], rps[-1]) if aal is not None else None}
    return out, summ


def class_edges(values: pd.Series, kind: str) -> list:
    """Bornes de classes pour la carte : ratio → fixes (10 / 30 / 60 %) ; euros → quartiles des valeurs non nulles."""
    if kind == "ratio":
        return [0.1, 0.3, 0.6]
    v = pd.to_numeric(values, errors="coerce")
    v = v[v > 0].dropna()
    if v.empty:
        return [1.0, 2.0, 3.0]
    e = sorted({float(x) for x in v.quantile([0.25, 0.5, 0.75])})
    while len(e) < 3:
        e.append(e[-1] * 2 if e[-1] > 0 else 1.0)
    return e[:3]


def damage_class(value, edges: list) -> Optional[int]:
    """0 : aucun dommage ; 1 à 4 selon les bornes ; None : pas de donnée."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    v = float(value)
    if v <= 0:
        return 0
    for i, e in enumerate(edges):
        if v < e:
            return i + 1
    return 4


# ───────────────────────────── chargement local JRC, légende de carte ─────────────────────────────
def merge_jrc(parts: list) -> dict:
    out = {"curves": {}, "maxdmg": {}, "notes": []}
    for p in parts:
        out["curves"].update(p.get("curves", {}))
        out["maxdmg"].update(p.get("maxdmg", {}))
        out["notes"].extend(p.get("notes", []))
    return out


def load_jrc_local() -> Optional[dict]:
    """Fichiers ``data/damage/jrc*`` (xlsx, csv) fusionnés ; ``None`` s'il n'y en a pas. Lève ``DamageError`` si illisibles."""
    files = sorted(p for p in DATA_DIR.glob("jrc*") if p.suffix.lower() in (".xlsx", ".xls", ".csv")) if DATA_DIR.exists() else []
    if not files:
        return None
    return merge_jrc([parse_jrc(p.read_bytes(), p.name) for p in files])


DMG_COLORS = {None: "#9e9e9e", 0: "#d9ead3", 1: "#fee391", 2: "#fe9929", 3: "#d95f0e", 4: "#993404"}


def _fmt_num(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") if v >= 100 else f"{v:,.2g}".replace(",", " ")


def legend_labels(edges: list, kind: str) -> dict:
    if kind == "ratio":
        p = [f"{round(e * 100)} %" for e in edges]
        return {0: "Aucun dommage", 1: f"< {p[0]}", 2: f"{p[0]} à < {p[1]}", 3: f"{p[1]} à < {p[2]}", 4: f"≥ {p[2]}"}
    p = [_fmt_num(e) + " €" for e in edges]
    return {0: "Aucune perte", 1: f"< {p[0]}", 2: f"{p[0]} à < {p[1]}", 3: f"{p[1]} à < {p[2]}", 4: f"≥ {p[2]}"}
