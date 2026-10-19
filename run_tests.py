"""Single-command test runner for ExoTransit-Conformal Engine."""

import os
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    print("=" * 70)
    print("Running ExoTransit-Conformal Engine Automated Test Suite...")
    print("=" * 70)
    retcode = pytest.main(["tests", "-v", "--tb=short"])
    sys.exit(retcode)
