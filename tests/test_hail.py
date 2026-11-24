import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from catmap import damage as dmg
from catmap import hail as hl

LATS = np.arange(40.0, 50.01, 0.25)           # 41 valeurs
LONS = np.arange(0.0, 10.01, 0.25)


def write_nc(path, var, z, attrs=None, extra=None):
    f = netcdf_file(str(path), "w")
    f.createDimension("lat", len(LATS))
    f.createDimension("lon", len(LONS))
    for n, v in (("lat", LATS), ("lon", LONS)):
        x = f.createVariable(n, "d", (n,))
        x[:] = v
    for name, arr in ([(var, z)] + list((extra or {}).items())):
        x = f.createVariable(name, "d", ("lat", "lon"))
        x[:] = arr
        for k, v in (attrs or {}).items():
            setattr(x, k, v)
    f.close()
    return Path(path)


def clim_field():
    z = np.full((len(LATS), len(LONS)), 0.1)
    z[:, :4] = 0.0                              # lon 0 à 0,75 : zéro
    return z


class Resp:
    def __init__(self, status=200, content=b"", js=None):
        self.status_code, self._c, self._js = status, content, js

    def iter_content(self, n):
        for i in range(0, len(self._c), n):
            yield self._c[i:i + n]

    def json(self):
        return self._js


class FakeZenodo:
    """Sert les NetCDF d'un dossier comme le ferait Zenodo (API des fichiers + téléchargement)."""
    def __init__(self, folder):
        self.folder, self.urls = Path(folder), []

    def get(self, url, **kw):
        self.urls.append(url)
        if url.startswith("https://zenodo.org/api/records/"):
            files = [{"key": p.name, "links": {"self": f"https://zenodo.org/files/{p.name}"}} for p in sorted(self.folder.glob("*.nc"))]
            return Resp(200, js={"files": files})
        name = url.split("/")[-1].split("?")[0]
        p = self.folder / name
        return Resp(200, p.read_bytes()) if p.exists() else Resp(404)


def build_source(folder, years=hl.YEARS, with_trend=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_nc(folder / hl.CLIM_FILE, "hail_ge5cm", clim_field() * 74, {"units": "events/yr"})   # cumul sur 74 ans, comme le vrai fichier
    if with_trend:
        tr = np.full_like(clim_field(), 0.05)             # +0,05 / décennie
        tr[:, 16:] = -0.04                                 # baisse à l'est (lon ≥ 4)
        pv = np.full_like(tr, 0.01)
        pv[:, 4:8] = 0.5                                   # non significatif au milieu
        write_nc(folder / hl.TREND_FILE, "trend_hail_ge5cm", tr, extra={"pvalue_hail_ge5cm": pv})
    for i, y in enumerate(years):
        z = clim_field() * (1 + (i % 5))                    # 1× à 5× la climatologie : séquence connue
        write_nc(folder / hl.annual_file(y), "prob_hail_ge5cm", z)


POINTS = pd.DataFrame({"name": ["a", "b", "c", "d"], "lat": ["45,0", 45.0, 45.0, 10.0], "long": ["5,0", 2.0, 0.25, 5.0]})
# a : lon 5 (clim 0,1, trend +) ; b : lon 2 (non significatif) ; c : lon 0,25 (clim 0) ; d : hors grille


class TestPrimitives(unittest.TestCase):
    def test_columns_cells_classes(self):
        self.assertEqual(hl.column_name(50), "grele_RP50_freq")
        self.assertEqual(hl.CLIM_COL, "grele_moy_freq")
        self.assertEqual(hl.cell_of(45.1, 5.13), (45.0, 5.25))
        self.assertEqual(hl.cell_of(10, 179.9)[1], -180.0)
        self.assertEqual([hl.freq_class(v) for v in (0.01, 0.1, 0.3, 0.9)], [0, 1, 2, 3])
        self.assertIsNone(hl.freq_class(float("nan")))

    def test_read_and_sample(self):
        with tempfile.TemporaryDirectory() as t:
            p = write_nc(Path(t) / "x.nc", "hail_ge5cm", clim_field())
            f = hl.field(hl.read_dataset(p), prefer=("hail_ge5cm",))
            self.assertEqual(f["name"], "hail_ge5cm")
            v = hl.sample(f, [45.0, 45.0, 10.0, 45.0], [5.0, 0.25, 5.0, 30.0])
            self.assertAlmostEqual(v[0], 0.1)
            self.assertAlmostEqual(v[1], 0.0)
            self.assertTrue(np.isnan(v[2]) and np.isnan(v[3]))      # hors grille

    def test_fill_value_and_longitude_0_360(self):
        arr = clim_field()
        arr[20, 20] = -9999.0
        with tempfile.TemporaryDirectory() as t:
            p = write_nc(Path(t) / "x.nc", "hail_ge5cm", arr, {"_FillValue": -9999.0})
            f = hl.field(hl.read_dataset(p))
            self.assertTrue(np.isnan(f["z"][20, 20]))
        f2 = {"lat": np.array([0.0, 0.25]), "lon": np.array([359.5, 359.75, 0.0]), "z": np.array([[1.0, 2.0, 3.0]] * 2)}
        self.assertEqual(hl.sample(f2, [0.0], [-0.25])[0], 2.0)

    def test_rp_levels_weibull_no_extrapolation(self):
        s = np.arange(1, 75, dtype=float)                           # 74 années : 1…74
        lv = hl.rp_levels(s, (2, 10, 50, 75, 100))
        self.assertAlmostEqual(lv[2], 37.5)                         # médiane
        self.assertAlmostEqual(lv[10], 0.9 * 75, places=6)          # quantile 0,9 : i/(n+1)=0,9 → i=67,5
        self.assertTrue(np.isnan(lv[100]))
        self.assertFalse(np.isnan(lv[75]))
        self.assertTrue(np.isnan(hl.rp_levels([1, 2], (2,))[2]))
        ec = hl.empirical_curve([1, 5, 3])
        self.assertEqual(list(ec["valeur"]), [5, 3, 1])
        self.assertAlmostEqual(ec["rp"].iloc[0], 4.0)

    def test_trend_factor(self):
        f = np.array([0.1, 0.1, 0.1, 0.0, 0.1])
        tr = np.array([0.05, -0.04, 0.05, 0.05, 0.05])               # par décennie
        p = np.array([0.01, 0.01, 0.5, 0.01, 0.01])
        fac = hl.trend_factor(f, tr, p, per_decade=True, years=38.0, p_max=0.05)
        self.assertGreater((0.1 + 0.005 * 38) / 0.1, 2.0)             # brut 2,9 → plafonné à 2
        self.assertEqual(fac[0], 2.0)
        self.assertAlmostEqual(fac[1], 0.5 + 0.0)                    # (0,1 − 0,004×38)/0,1 = −0,52 → plancher 0,5
        self.assertEqual(fac[2], 1.0)                                # non significatif
        self.assertEqual(fac[3], 1.0)                                # fréquence nulle
        self.assertEqual(hl.trend_factor(f, tr, p, p_max=0.05, rel_min_pct=1000)[0], 1.0)      # seuil relatif trop haut
        fac_y = hl.trend_factor([1.0], [0.01], [0.0], per_decade=False, years=10, p_max=0.05)  # +0,01/an × 10 = +10 %
        self.assertAlmostEqual(fac_y[0], 1.1)
        self.assertGreater(len(hl.threshold_table(f, tr, p)), 5)


class TestEstimate(unittest.TestCase):
    def test_clim_mode(self):
        with tempfile.TemporaryDirectory() as t:
            src, cache = Path(t) / "src", Path(t) / "cache"
            build_source(src, years=())
            sess = FakeZenodo(src)
            out, s = hl.estimate(POINTS, "lat", "long", "clim", session=sess, cache_dir=cache, data_dir=Path(t) / "nodata")
            self.assertEqual(s["points"], 4)
            self.assertEqual(s["ok"], 3)               # a, b, c lus ; c = 0 est une valeur valide
            self.assertEqual(s["sans_donnee"], 1)
            self.assertEqual(list(out[hl.CLIM_COL].round(3))[:3], [0.1, 0.1, 0.0])
            self.assertTrue(np.isnan(out[hl.CLIM_COL].iloc[3]))
            self.assertEqual(out["grele_statut"].iloc[0], "ok")
            self.assertIn("sans donnée", out["grele_statut"].iloc[3])

    def test_clim_with_trend(self):
        pts = pd.DataFrame({"lat": [45.0, 45.0, 45.0], "long": [5.0, 1.0, 2.0]})      # baisse signif. ; p = 0,5 ; hausse signif.
        with tempfile.TemporaryDirectory() as t:
            src, cache = Path(t) / "src", Path(t) / "cache"
            build_source(src, years=())
            out, s = hl.estimate(pts, "lat", "long", "clim", use_trend=True, trend_opts={"per_decade": True, "years": 38.0, "p_max": 0.05},
                                 session=FakeZenodo(src), cache_dir=cache, data_dir=Path(t) / "nodata")
            self.assertEqual(list(out["grele_facteur_tendance"]), [0.5, 1.0, 2.0])
            self.assertAlmostEqual(out[hl.CLIM_COL].iloc[2], 0.2)
            self.assertAlmostEqual(out[hl.CLIM_COL + "_brut"].iloc[2], 0.1)
            self.assertAlmostEqual(out[hl.CLIM_COL].iloc[0], 0.05)
            self.assertEqual((s["hausse"], s["baisse"]), (1, 1))

    def test_annual_mode_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            src, cache = Path(t) / "src", Path(t) / "cache"
            build_source(src, with_trend=False)
            sess = FakeZenodo(src)
            kw = dict(session=sess, cache_dir=cache, data_dir=Path(t) / "nodata")
            out, s = hl.estimate(POINTS, "lat", "long", "annual", rps=(2, 10), budget_s=0.0, **kw)    # budget nul : rien ne se télécharge
            self.assertEqual(s["en_attente"], 74)
            self.assertTrue(out["grele_RP2_freq"].isna().all())
            self.assertEqual(out["grele_statut"].iloc[0], "en attente (années à télécharger)")
            out, s = hl.estimate(POINTS, "lat", "long", "annual", rps=(2, 10, 200), **kw)
            self.assertEqual(s["en_attente"], 0)
            self.assertEqual(s["rps"], [2, 10])         # 200 > 75 ans : écartée
            # valeurs annuelles : clim × (1…5) répétées ; médiane = 3 × 0,1 ; RP10 supérieur
            self.assertAlmostEqual(out["grele_RP2_freq"].iloc[0], 0.3, places=2)
            self.assertGreater(out["grele_RP10_freq"].iloc[0], out["grele_RP2_freq"].iloc[0])
            self.assertEqual(out["grele_RP2_freq"].iloc[2], 0.0)
            self.assertEqual(s["ok"], 3)
            self.assertFalse(list(cache.glob("*.nc")))   # gros fichiers supprimés, seules les petites séries restent
            n_calls = len(sess.urls)
            hl.estimate(POINTS, "lat", "long", "annual", rps=(2, 10), **kw)
            self.assertEqual(len(sess.urls), n_calls)    # tout est en cache : aucun nouveau téléchargement
            ser = hl.load_series([hl.cell_of(45, 5)], cache_dir=cache)
            self.assertEqual(len(next(iter(ser.values()))), 74)
            self.assertIn(hl._cell_key(hl.cell_of(45, 5)), hl.series_cache_cells(cache))

    def test_new_cell_redownloads_only_missing_years(self):
        with tempfile.TemporaryDirectory() as t:
            src, cache = Path(t) / "src", Path(t) / "cache"
            build_source(src, years=hl.YEARS[:4], with_trend=False)
            kw = dict(session=FakeZenodo(src), cache_dir=cache, data_dir=Path(t) / "nodata")
            r1 = hl.ensure_years([hl.cell_of(45, 5)], hl.YEARS[:4], **kw)
            self.assertEqual(r1["pending"], 0)
            r2 = hl.ensure_years([hl.cell_of(45, 5), hl.cell_of(46, 6)], hl.YEARS[:4], **kw)
            self.assertEqual(r2["pending"], 0)
            self.assertEqual(len(hl.load_series([hl.cell_of(46, 6)], hl.YEARS[:4], cache)[hl._cell_key(hl.cell_of(46, 6))]), 4)

    def test_download_error_is_reported(self):
        with tempfile.TemporaryDirectory() as t:
            class Bad:
                def get(self, url, **kw):
                    return Resp(200, b"<html>erreur</html>", {"files": []}) if "api/records" not in url else Resp(500)
            r = hl.ensure_years([hl.cell_of(45, 5)], hl.YEARS[:5], session=Bad(), cache_dir=Path(t) / "c", data_dir=Path(t) / "d")
            self.assertEqual(r["pending"], 5)
            self.assertIn("ni NetCDF ni HDF5", next(iter(r["errors"].values())))
            self.assertLessEqual(len(r["errors"]), 5)          # 3 erreurs déclenchent l'arrêt ; les téléchargements déjà lancés peuvent finir

    def test_local_data_dir_has_priority_and_is_kept(self):
        with tempfile.TemporaryDirectory() as t:
            data = Path(t) / "data"
            build_source(data, years=hl.YEARS[:2], with_trend=False)
            r = hl.ensure_years([hl.cell_of(45, 5)], hl.YEARS[:2], session=None, cache_dir=Path(t) / "c", data_dir=data)
            self.assertEqual(r["pending"], 0)
            self.assertTrue((data / hl.annual_file(1950)).exists())         # fichier de l'utilisateur jamais supprimé

    def test_invalid_mode(self):
        with self.assertRaises(hl.HailError):
            hl.estimate(POINTS, "lat", "long", "zzz")


class TestCompact(unittest.TestCase):
    def build(self, t):
        src = Path(t) / "src"
        build_source(src)
        data = Path(t) / "data"
        data.mkdir()
        info = hl.build_compact(src, data / hl.COMPACT_NAME, progress=lambda *_: None)
        return src, data, info

    def test_layers_match_point_by_point_quantiles(self):
        rng = np.random.default_rng(3)
        A = rng.gamma(0.5, 0.2, (74, 6, 7)).astype(np.float32)
        A[:, 0, 0] = np.nan
        lay = hl.rp_layers(A, (2, 10, 50, 75, 100))
        for (i, j) in ((1, 2), (5, 6), (3, 0)):
            ref = hl.rp_levels(A[:, i, j], (2, 10, 50, 75, 100))
            for rp in (2, 10, 50, 75):
                self.assertAlmostEqual(float(lay[rp][i, j]), ref[rp], places=5)
        self.assertTrue(np.isnan(lay[100]).all())
        self.assertTrue(np.isnan(lay[2][0, 0]))

    def test_build_and_use_without_network(self):
        with tempfile.TemporaryDirectory() as t:
            src, data, info = self.build(t)
            self.assertEqual(info["annees"], 74)
            self.assertLess(info["taille_mo"], 1)

            class NoNet:
                def get(self, *a, **k):
                    raise AssertionError("aucun accès réseau attendu")
            kw = dict(session=NoNet(), cache_dir=Path(t) / "cache", data_dir=data)
            out, s = hl.estimate(POINTS, "lat", "long", "annual", rps=(2, 10, 200), use_trend=True, **kw)
            self.assertEqual(s["source"], "fichier précalculé")
            self.assertEqual((s["en_attente"], s["rps"]), (0, [2, 10]))
            self.assertAlmostEqual(out["grele_RP2_freq_brut"].iloc[0], 0.3, places=2)
            self.assertAlmostEqual(out["grele_moy_freq"].iloc[0], 0.3, delta=0.02)          # moyenne des 74 années (1…5 × 0,1)
            self.assertEqual(out["grele_RP2_freq_brut"].iloc[2], 0.0)
            self.assertTrue(np.isnan(out["grele_RP2_freq"].iloc[3]))                         # hors grille
            # identique au calcul direct à partir des fichiers annuels
            direct, _ = hl.estimate(POINTS, "lat", "long", "annual", rps=(2, 10), session=FakeZenodo(src), cache_dir=Path(t) / "c2",
                                    data_dir=Path(t) / "nodata")
            for c in ("grele_RP2_freq", "grele_RP10_freq"):
                self.assertTrue(np.allclose(out[c + "_brut"].iloc[:3], direct[c].iloc[:3], rtol=2e-3, atol=1e-4), c)
            rows = hl.diagnose(NoNet(), Path(t) / "c", data)
            self.assertTrue([r for r in rows if r["étape"] == "fichier précalculé" and r["résultat"] == "OK"])
            o2, _ = hl.estimate(POINTS, "lat", "long", "clim", **kw)
            self.assertAlmostEqual(o2["grele_moy_freq"].iloc[0], 0.3, delta=0.02)
            with self.assertRaises(hl.HailError):
                hl.estimate(POINTS, "lat", "long", "annual", rps=(3,), **kw)                  # RP absente du fichier

    def test_raw_data_ignores_compact_file(self):
        with tempfile.TemporaryDirectory() as t:
            src, data, _ = self.build(t)
            sess = FakeZenodo(src)
            out, s = hl.estimate(POINTS, "lat", "long", "clim", session=sess, cache_dir=Path(t) / "c", data_dir=data, use_compact=False)
            self.assertEqual(s["source"], "Zenodo")
            self.assertTrue(sess.urls)                                   # le web a bien été interrogé
            self.assertAlmostEqual(out["grele_moy_freq"].iloc[0], 0.1)    # cumul ÷ 74

    def test_file_found_in_parent_data_folder(self):
        with tempfile.TemporaryDirectory() as t:
            data = Path(t) / "data"
            (data / "grele").mkdir(parents=True)
            self.assertIsNone(hl.compact_path(data / "grele"))
            build_source(Path(t) / "src")
            hl.build_compact(Path(t) / "src", data / hl.COMPACT_NAME, progress=lambda *_: None)      # déposé dans data/ et non data/grele/
            self.assertEqual(hl.compact_path(data / "grele"), data / hl.COMPACT_NAME)

    def test_missing_files_and_too_big(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(hl.HailError):
                hl.build_compact(t, Path(t) / "x.npz", progress=lambda *_: None)
            src = Path(t) / "src"
            build_source(src)
            with self.assertRaises(hl.HailError):
                hl.build_compact(src, Path(t) / "y.npz", max_mb=0.000001, progress=lambda *_: None)

    def test_incomplete_file_is_reported(self):
        with tempfile.TemporaryDirectory() as t:
            np.savez(Path(t) / hl.COMPACT_NAME, lat=np.zeros(2))
            with self.assertRaises(hl.HailError):
                hl.compact_fields(Path(t) / hl.COMPACT_NAME)


class TestParallelDownload(unittest.TestCase):
    def test_parallel_years_and_direct_url_first(self):
        with tempfile.TemporaryDirectory() as t:
            src, cache = Path(t) / "src", Path(t) / "cache"
            build_source(src, with_trend=False)
            sess = FakeZenodo(src)
            r = hl.ensure_years([hl.cell_of(45, 5)], cache_dir=cache, data_dir=Path(t) / "nodata", session=sess, workers=4)
            self.assertEqual(r["pending"], 0)
            self.assertFalse([u for u in sess.urls if "api/records" in u])      # pas d'appel à l'API Zenodo si l'adresse directe répond
            self.assertEqual(len([u for u in sess.urls if u.endswith("download=1")]), 74)
            self.assertFalse(list(cache.glob("*.part")) or list(cache.glob("*.nc")))


class TestDamages(unittest.TestCase):
    def _base(self, mode):
        with tempfile.TemporaryDirectory() as t:
            src, cache = Path(t) / "src", Path(t) / "cache"
            build_source(src, with_trend=False)
            return hl.estimate(POINTS, "lat", "long", mode, rps=(2, 10), session=FakeZenodo(src), cache_dir=cache, data_dir=Path(t) / "nodata")

    def test_climatology_damage(self):
        base, s = self._base("clim")
        classes = pd.Series("commercial", index=base.index)
        values = pd.Series([1_000_000.0] * 4)
        o, sd = hl.estimate_damages(base, s, 0.01, classes, values)
        self.assertAlmostEqual(o["dommage_grele_PAM_eur"].iloc[0], 0.1 * 0.01 * 1e6)       # 1 000 €
        self.assertEqual(o["dommage_grele_PAM_eur"].iloc[2], 0.0)
        self.assertTrue(np.isnan(o["dommage_grele_PAM_eur"].iloc[3]))
        self.assertAlmostEqual(sd["pam_totale"], 1000.0 * 2)
        self.assertEqual(sd["sans_hauteur"], 1)

    def test_annual_damage_and_aal(self):
        base, s = self._base("annual")
        classes = pd.Series("commercial", index=base.index)
        values = pd.Series([1_000_000.0] * 4)
        o, sd = hl.estimate_damages(base, s, 0.01, classes, values)
        self.assertIn("dommage_grele_RP2_eur", o.columns)
        self.assertIn("dommage_grele_RP10_eur", o.columns)
        self.assertIn("dommage_grele_PAM_eur", o.columns)
        r2 = base["grele_RP2_freq"].iloc[0]
        self.assertAlmostEqual(o["dommage_grele_RP2_eur"].iloc[0], r2 * 0.01 * 1e6, places=3)
        # PAM = trapèze entre 1/10 et 1/2
        r10 = base["grele_RP10_freq"].iloc[0]
        exp = 0.5 * (0.01 * r2 * 1e6 + 0.01 * r10 * 1e6) * (0.5 - 0.1)
        self.assertAlmostEqual(o["dommage_grele_PAM_eur"].iloc[0], exp, places=3)

    def test_ratio_capped_and_validation(self):
        c = hl.event_curve(0.5)
        self.assertEqual(float(c.at([10.0])[0]), 1.0)
        self.assertAlmostEqual(float(c.at([1.0])[0]), 0.5)
        with self.assertRaises(ValueError):
            hl.event_curve(0.0)
        base, s = self._base("clim")
        with self.assertRaises(dmg.DamageError):
            hl.estimate_damages(base, s, 2.0, pd.Series("commercial", index=base.index), pd.Series([1.0] * 4))


class TestDiagnose(unittest.TestCase):
    def test_diagnose_reports_everything(self):
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "src"
            build_source(src, years=(2000,))
            rows = hl.diagnose(FakeZenodo(src), Path(t) / "c", Path(t) / "nodata")
            txt = " ".join(r["détail"] for r in rows)
            self.assertIn("hail_ge5cm", txt)
            self.assertIn("trend_hail_ge5cm", txt)
            self.assertIn("units = events/yr", txt)
            self.assertFalse([r for r in rows if r["résultat"] == "ÉCHEC"])
            self.assertTrue(any(r["étape"].startswith("cohérence") for r in rows))

    def test_diagnose_failure_is_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            class Down:
                def get(self, url, **kw):
                    return Resp(503)
            rows = hl.diagnose(Down(), Path(t) / "c", Path(t) / "d")
            self.assertTrue([r for r in rows if r["résultat"] == "ÉCHEC"])


if __name__ == "__main__":
    unittest.main()
