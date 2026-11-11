"""Run privacy-preserving regression tests. No model download required."""
import unittest

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromNames(['test_review_flow', 'test_ui_flow'])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
