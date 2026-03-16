"""
Compatibility shim: older imports may reference scurve.features.pref_fit.
The implementation lives in scurve.features.pre_fit.
"""
from __future__ import annotations

from scurve.features.pre_fit import *  # noqa: F401,F403