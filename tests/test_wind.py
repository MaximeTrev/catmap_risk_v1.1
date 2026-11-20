import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from catmap import wind, damage as dmg


def _csv(lat=48.0, lon=2.0, n=24 * 3, base="2000-01-01", gust=12.0, peak_hour=30, peak=33.0):
    t = pd.date_range(base, periods=n, freq="h")
    v = np.full(n, gust)
    v[peak_hour] = peak
    return pd.DataFrame({"valid_time": t.strftime("%Y-%m-%d %H:%M:%S"), "latitude": lat, "longitude": lon, "fg10": v}).to_csv(index=False).encode()


class FakeClient:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def retrieve(self, dataset, request, target):
        self.calls.append((dataset, request))
        if self.fail and self.fail(dataset, request):
            raise RuntimeError("request too large, cost limit exceeded")
        if dataset == wind.TS_DATASET:
            y0 = int(request["date"][0][:4])
            Path(target).write_bytes(_csv(base=f"{y0}-01-01"))
        else:
            Path(target).write_bytes(b"track_id,time,lon,lat,gust\ns1,2000-01-02 00:00,2,48,30\ns1,2000-01-02 06:00,3,49,35\n")
        return target


class TestBasics(unittest.TestCase):
    def test_cell_and_columns(self):
        self.assertEqual(wind.cell_of(48.1, 2.13), (48.0, 2.25))
        self.assertEqual(wind.cell_of(10, 190), (10.0, -170.0))
        self.assertEqual(wind.column_name("vent", 100), "vent_RP100_gust")
        self.assertEqual(wind.column_name("tempete", 50), "tempete_RP50_gust")

    def test_parse_csv_zip_and_errors(self):
        s = wind.parse_gust_series(_csv())
        self.assertAlmostEqual(s.max(), 33.0)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a.csv", _csv())
        self.assertAlmostEqual(wind.parse_gust_series(buf.getvalue()).max(), 33.0)
        with self.assertRaises(wind.WindError) as cm:
            wind.parse_gust_series(b"a,b\n1,2\n")
        self.assertIn("colonnes", str(cm.exception))

    def test_daily_and_annual_maxima(self):
        d = wind.daily_max(wind.parse_gust_series(_csv()))
        self.assertEqual(len(d), 3)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.npz"
            wind.save_daily(p, d)
            self.assertEqual(len(wind.load_daily(p)), 3)
        idx = pd.date_range("2001-01-01", "2003-12-31", freq="D")
        am = wind.annual_maxima(pd.Series(np.arange(len(idx), dtype=float), index=idx))
        self.assertEqual(list(am.index), [2001, 2002, 2003])


class TestExtremes(unittest.TestCase):
    def test_gev_recovers_gumbel(self):
        x = np.random.default_rng(1).gumbel(25, 3, 20000)
        p = wind.gev_fit(x)
        self.assertAlmostEqual(wind.gev_return_level(p, 100), 25 + 3 * np.log(100), delta=1.0)

    def test_annual_levels_need_years(self):
        a = pd.Series(np.linspace(20, 30, 10), index=range(2000, 2010))
        self.assertIsNone(wind.return_levels_annual(a, (100,))[100])
        a = pd.Series(np.random.default_rng(2).gumbel(25, 3, 40), index=range(1980, 2020))
        lv = wind.return_levels_annual(a, (10, 100))
        self.assertLess(lv[10], lv[100])

    def test_storm_levels_monotonic_and_none(self):
        pk = 20 + np.random.default_rng(3).exponential(4, 400)
        lv = wind.return_levels_storms(pk, 40, (2, 10, 100))
        self.assertTrue(lv[2] < lv[10] < lv[100])
        self.assertIsNone(wind.return_levels_storms(pk[:5], 40, (10,))[10])


class TestCdsFlow(unittest.TestCase):
    def test_requests_use_real_parameter_names(self):
        r = wind.build_timeseries_request(48.0, 2.0, 1990, 1999)
        self.assertEqual(r["variable"], ["10m_wind_gust_since_previous_post_processing"])
        self.assertEqual(r["data_format"], "csv")
        self.assertEqual(r["date"], ["1990-01-01/1999-12-31"])
        t = wind.build_tracks_request("hodges", 2000, 2001)
        self.assertEqual((t["product"], t["tracking_algorithm"]), ("windstorm_track", ["hodges"]))
        self.assertEqual(t["year"], ["2000", "2001"])

    def test_ensure_cells_cache_and_budget(self):
        with tempfile.TemporaryDirectory() as t:
            cli = FakeClient()
            cells = [(48.0, 2.0), (49.0, 3.0)]
            r = wind.ensure_cells(cells, 2000, 2000, cli, Path(t))
            self.assertEqual(len(r["done"]), 2)
            n = len(cli.calls)
            r = wind.ensure_cells(cells, 2000, 2000, cli, Path(t))
            self.assertEqual(len(cli.calls), n)                              # cache réutilisé
            r = wind.ensure_cells([(10.0, 10.0)], 2000, 2000, cli, Path(t), budget_s=-1)
            self.assertEqual(r["pending"], [(10.0, 10.0)])                   # budget épuisé : en attente

    def test_ensure_cells_reports_errors_and_splits(self):
        with tempfile.TemporaryDirectory() as t:
            span = lambda rq: int(rq["date"][0].split("/")[1][:4]) - int(rq["date"][0][:4])  # noqa: E731
            cli = FakeClient(fail=lambda d, rq: d == wind.TS_DATASET and span(rq) > 30)
            r = wind.ensure_cells([(48.0, 2.0)], 1950, 2023, cli, Path(t))
            self.assertEqual(len(r["done"]), 1, r["errors"])                 # découpage automatique
            self.assertGreater(len(cli.calls), 1)
            always = FakeClient(fail=lambda d, rq: True)
            r = wind.ensure_cells([(1.0, 1.0)], 2000, 2001, always, Path(t))
            self.assertEqual(len(r["errors"]), 1)

    def test_tracks_storms_and_losses(self):
        with tempfile.TemporaryDirectory() as t:
            tr = wind.fetch_tracks(FakeClient(), "hodges", 2000, 2000, Path(t))
            self.assertEqual(len(tr), 2)
            st = wind.extract_storms(tr, [(48.0, 2.0)], radius_km=300)
            self.assertEqual(len(st), 1)
            self.assertEqual(st.loc[0, "peak_lat"], 49.0)                    # point de la trajectoire à la rafale maximale
            far = wind.extract_storms(tr, [(0.0, 100.0)], radius_km=300)
            self.assertTrue(far.empty)
            d = wind.daily_max(wind.parse_gust_series(_csv(base="2000-01-01")))
            self.assertAlmostEqual(wind.storm_gust(d, st.loc[0, "peak_time"]), 33.0)
            G = np.array([[40.0, 40.0], [10.0, 10.0]])
            curve = dmg.emanuel_curve()
            loss = wind.event_losses(G, ["residential", "residential"], [100.0, 200.0], {"residential": curve})
            self.assertGreater(loss[0], 0)
            self.assertEqual(loss[1], 0)

    def test_compute_and_names(self):
        idx = pd.date_range("1990-01-01", "2023-12-31", freq="D")
        d = pd.Series(np.random.default_rng(5).gumbel(18, 3, len(idx)).astype("float32"), index=idx)
        v = wind.compute_vent({(48.0, 2.0): d}, (10, 100))
        self.assertLess(v[(48.0, 2.0)][10], v[(48.0, 2.0)][100])
        self.assertEqual(wind.known_name("1999-12-26 12:00"), "Lothar")
        self.assertEqual(wind.known_name("2001-05-01"), "")
        self.assertEqual(wind.gust_class(30.0), 2)
        self.assertIsNone(wind.gust_class(float("nan")))

    def test_tracks_fall_back_to_simpler_request(self):
        class Strict(FakeClient):
            def retrieve(self, dataset, request, target):
                if dataset == wind.ST_DATASET and ("month" in request or "day" in request):
                    self.calls.append((dataset, request))
                    raise RuntimeError("400 Client Error: Bad Request ... invalid request Request has not produced a valid combination of values")
                return super().retrieve(dataset, request, target)
        class Strict400(FakeClient):
            def retrieve(self, dataset, request, target):
                raise RuntimeError("400 Client Error: Bad Request invalid request")
        with tempfile.TemporaryDirectory() as t:
            tr = wind.fetch_tracks(Strict(), "hodges", 2000, 2000, Path(t))
            self.assertEqual(len(tr), 2)
            with self.assertRaises(wind.WindError) as cm:
                wind.fetch_tracks(Strict400(), "hodges", 2001, 2001, Path(t))
            self.assertIn("variante 3", str(cm.exception))

    def test_parallel_cells(self):
        with tempfile.TemporaryDirectory() as t:
            cells = [(40.0 + i, 2.0) for i in range(9)]
            r = wind.ensure_cells(cells, 2000, 2000, FakeClient(), Path(t), workers=4)
            self.assertEqual(sorted(r["done"]), sorted(cells))

    def test_queue_full_is_pending_not_error(self):
        class Full(FakeClient):
            def retrieve(self, dataset, request, target):
                raise RuntimeError("400 Bad Request The job has been rejected Maximum number queued requests for this dataset has been reached. "
                                   "Please configure your scripts to request data sequentially")
        with tempfile.TemporaryDirectory() as t:
            r = wind.ensure_cells([(40.0, 2.0), (41.0, 2.0)], 2000, 2000, Full(), Path(t))
            self.assertTrue(r["queue_full"])
            self.assertEqual(len(r["pending"]), 2)
            self.assertFalse(r["errors"])

    def test_tracks_summary(self):
        tr = wind.parse_tracks(b"track_id,time,lon,lat,gust\ns1,2000-01-02 00:00,2,48,30\ns2,2000-01-03 00:00,100,10,35\n")
        sm = wind.tracks_summary(tr)
        self.assertEqual((sm["tempetes"], sm["tempetes_dans_domaine"]), (2, 1))

    def test_emanuel(self):
        c = dmg.emanuel_curve()
        self.assertEqual(float(c.at(20)), 0.0)
        self.assertAlmostEqual(float(c.at(75)), 0.5, delta=0.02)
        with self.assertRaises(dmg.DamageError):
            dmg.emanuel_curve(50, 40)


if __name__ == "__main__":
    unittest.main()
