"""Single-command test runner for ExoTransit-Conformal Engine."""

import sys
import pytest

if __name__ == "__main__":
    print("=" * 70)
    print("Running ExoTransit-Conformal Engine Automated Test Suite...")
    print("=" * 70)
    retcode = pytest.main(["tests/", "-v", "--tb=short"])
    sys.exit(retcode)
