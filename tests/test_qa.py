import unittest
import numpy as np
import pandas as pd
from catmap.qa import make_sample, precision_report, wilson_interval


def sites(n=1000):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"name": [f"s{i}" for i in range(n)], "lat": rng.uniform(40, 50, n), "long": rng.uniform(0, 10, n),
                         "pays": rng.choice(["France"] * 8 + ["Allemagne"] * 2 + ["Monaco"], n)})


class TestQA(unittest.TestCase):
    def test_reproducible_and_sized(self):
        a, b = make_sample(sites(), 100, seed=7), make_sample(sites(), 100, seed=7)
        self.assertEqual(list(a.index), list(b.index)); self.assertEqual(len(a), 100)
        self.assertNotEqual(list(a.index), list(make_sample(sites(), 100, seed=8).index))

    def test_stratified_keeps_small_strata_and_links(self):
        s = make_sample(sites(), 50, seed=1, strata=["pays"])
        self.assertIn("Monaco", set(s.pays)); self.assertLessEqual(len(s), 50)
        self.assertTrue(s.lien_google_maps.str.startswith("https://www.google.com/maps").all())

    def test_small_population(self):
        self.assertEqual(len(make_sample(sites(10), 500)), 10)
        self.assertTrue(make_sample(sites(0).iloc[0:0], 5).empty)

    def test_wilson(self):
        lo, hi = wilson_interval(90, 100)
        self.assertAlmostEqual(lo, 0.826, places=2); self.assertAlmostEqual(hi, 0.945, places=2)
        self.assertTrue(np.isnan(wilson_interval(0, 0)[0]))

    def test_report(self):
        df = pd.DataFrame({"precision": [1, 1, 1, 0, 2, None], "pays": ["FR", "FR", "FR", "DE", "DE", "DE"]})
        r = precision_report(df).iloc[0]
        self.assertEqual((r.bonnes, r.mauvaises, r.incertaines), (3, 1, 1)); self.assertAlmostEqual(r.precision, 0.75)
        self.assertAlmostEqual(r.precision_min, 0.6); self.assertAlmostEqual(r.precision_max, 0.8)
        self.assertEqual(len(precision_report(df, "pays")), 2)


if __name__ == "__main__":
    unittest.main()
