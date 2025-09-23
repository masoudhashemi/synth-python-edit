"""Convenience script to execute the package's unit tests."""

import sys
import unittest


def main() -> int:
    suite = unittest.defaultTestLoader.discover("synthetic_debug/tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
