import unittest

from catmap.references import REFS, refs_md


class TestReferences(unittest.TestCase):
    def test_blocks(self):
        for k in REFS:
            self.assertIn("•", refs_md(k))
        self.assertEqual(refs_md("inconnue"), "")
        self.assertEqual(refs_md("gev", "gev").count("Hosking"), 1)          # pas de doublon


if __name__ == "__main__":
    unittest.main()
