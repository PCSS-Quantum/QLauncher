"""Import managing for problem formulations (attempt to import as much as possible)."""

import contextlib

from .bqm import *
from .qubo import *

with contextlib.suppress(ImportError):
	from .hamiltonian import *
