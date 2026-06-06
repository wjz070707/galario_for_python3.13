from __future__ import annotations

import numpy as np

from .. import _nanobind_double_cuda as _raw
from ..nanobind_api_v2 import build_api

globals().update(build_api(_raw, __name__, np.float64, np.complex128))
