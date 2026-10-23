"""Lets the tests import the tzaar package, which lives in the tzaar/ folder next to this one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tzaar"))
