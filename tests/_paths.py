"""Shared path setup so the tests can be run from anywhere."""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
FIXTURES = os.path.join(TESTS_DIR, "fixtures")
TMP = os.path.join(TESTS_DIR, "_tmp")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.makedirs(TMP, exist_ok=True)
