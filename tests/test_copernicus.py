import unittest

from catmap import copernicus as cop


class R:
    def __init__(self, status=200, data=None):
        self.status_code, self._d = status, data

    def json(self):
        if self._d is None:
            raise ValueError("no json")
        return self._d


class S:
    def __init__(self, r):
        self.r, self.urls = r, []

    def get(self, url, **k):
        self.urls.append(url)
        return self.r


FORM = [
    {"name": "variable", "label": "Variable", "type": "StringListArrayWidget",
     "details": {"groups": [{"values": ["10m_wind_gust_since_previous_post_processing", "2m_temperature"],
                             "labels": {"10m_wind_gust_since_previous_post_processing": "10m wind gust since previous post-processing"}}]}},
    {"name": "data_format", "label": "Data format", "type": "StringChoiceWidget", "details": {"values": ["csv", "netcdf"]}},
    {"name": "location", "label": "Location", "type": "GeographicLocationWidget", "details": {}},
]


class TestCopernicus(unittest.TestCase):
    def test_summarize_finds_gust_and_formats(self):
        rows = cop.summarize_form(FORM)
        by = {r["champ (API)"]: r for r in rows}
        self.assertIn("10m_wind_gust_since_previous_post_processing", by["variable"]["valeurs « gust »"])
        self.assertEqual(by["data_format"]["nb valeurs"], 2)
        self.assertEqual(by["location"]["nb valeurs"], 0)

    def test_summarize_dict_form_and_garbage(self):
        self.assertEqual(len(cop.summarize_form({"widgets": FORM})), 3)
        self.assertEqual(cop.summarize_form("x"), [])

    def test_fetch_form_errors(self):
        s = S(R(200, FORM))
        self.assertEqual(cop.fetch_form("ds", s), FORM)
        self.assertIn("/collections/ds/form.json", s.urls[0])
        with self.assertRaises(cop.CopernicusError):
            cop.fetch_form("ds", S(R(404)))
        with self.assertRaises(cop.CopernicusError):
            cop.fetch_form("ds", S(R(200, None)))

    def test_key_present_never_returns_the_key(self):
        self.assertTrue(cop.key_present({"CDSAPI_KEY": "abc"}))
        self.assertFalse(cop.key_present({}, {}))
        self.assertTrue(cop.key_present({}, {"CDS_API_KEY": "x"}))


if __name__ == "__main__":
    unittest.main()
