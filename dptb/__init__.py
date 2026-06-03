"""Compatibility shim for historical checkpoint/module paths.

EMolES is now the primary package. This module keeps legacy ``dptb`` imports
resolvable while downstream code migrates to ``emoles``.
"""

import importlib
import sys

_emoles = importlib.import_module("emoles")
sys.modules[__name__] = _emoles
