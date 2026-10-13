import math
import unittest
from catmap import geo

A, F = 6378137.0, 1 / 298.257223563
E2 = F * (2 - F)


def expected_area(lat0, d):
    phi = math.radians(lat0); s = math.sin(phi)
    N = A / math.sqrt(1 - E2 * s * s); M = A * (1 - E2) / (1 - E2 * s * s) ** 1.5
    return (M * math.radians(d)) * (N * math.cos(phi) * math.radians(d))


class TestArea(unittest.TestCase):
    def test_area_matches_ellipsoid(self):
        for lat in (0, 48.8566, -33.87, 64.1, 78.2):
            d = 0.001
            ring = [(10, lat - d / 2), (10 + d, lat - d / 2), (10 + d, lat + d / 2), (10, lat + d / 2)]
            self.assertAlmostEqual(geo.ring_area_m2(ring) / expected_area(lat, d), 1.0, places=5)

    def test_antimeridian(self):
        r = [(179.9995, 10), (-179.9995, 10), (-179.9995, 10.001), (179.9995, 10.001)]
        self.assertAlmostEqual(geo.ring_area_m2(r) / expected_area(10.0005, 0.001), 1.0, places=4)

    def test_degenerate(self):
        self.assertEqual(geo.ring_area_m2([(0, 0), (1, 1)]), 0.0)


class TestPoints(unittest.TestCase):
    def test_representative_point_inside_concave(self):
        for poly in ([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)],
                     [(0, 0), (9, 0), (9, 10), (6, 10), (6, 2), (3, 2), (3, 10), (0, 10)]):
            c = geo.ring_centroid(poly)
            self.assertFalse(geo.point_in_ring(*c, poly))      # le centroïde est hors du bâti (H7)
            rp = geo.representative_point((poly, []))
            self.assertTrue(geo.point_in_ring(*rp, poly))

    def test_haversine(self):
        self.assertAlmostEqual(geo.haversine_m(0, 0, 0, 1), 111195, delta=200)


class TestMultipolygon(unittest.TestCase):
    def mp(self):
        outer = [{"type": "way", "role": "outer", "geometry": [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 0.002}, {"lat": 0.002, "lon": 0.002}]},
                 {"type": "way", "role": "outer", "geometry": [{"lat": 0.002, "lon": 0.002}, {"lat": 0.002, "lon": 0}, {"lat": 0, "lon": 0}]}]
        inner = [{"type": "way", "role": "inner", "geometry": [{"lat": 0.0005, "lon": 0.0005}, {"lat": 0.0005, "lon": 0.001},
                 {"lat": 0.001, "lon": 0.001}, {"lat": 0.001, "lon": 0.0005}, {"lat": 0.0005, "lon": 0.0005}]}]
        return outer, inner

    def test_ring_assembly_and_hole(self):
        outer, inner = self.mp()
        polys, inc = geo.build_multipolygon(outer + inner)
        self.assertEqual((len(polys), len(polys[0][1]), inc), (1, 1, 0))
        full = geo.ring_area_m2(polys[0][0]); hole = geo.ring_area_m2(polys[0][1][0])
        self.assertAlmostEqual(geo.polygon_area_m2(polys[0]), full - hole, places=3)
        self.assertLess(hole, full)

    def test_missing_member_flagged(self):
        outer, _ = self.mp()
        polys, inc = geo.build_multipolygon(outer[:1])
        self.assertEqual((polys, inc), ([], 1))

    def test_reversed_segment(self):
        a = [(0, 0), (1, 0), (1, 1)]; b = [(0, 0), (0, 1), (1, 1)]  # b à l'envers
        rings, op = geo.assemble_rings([a, b])
        self.assertEqual((len(rings), len(op)), (1, 0))


if __name__ == "__main__":
    unittest.main()
