from typing import Literal

from qlauncher.base import Backend
from qlauncher.exceptions import DependencyError

try:
	from qoolqit.execution import LocalEmulator
except ImportError as e:
	raise DependencyError(e, install_hint='pasqal', private=False) from e


class PasqalBackend(Backend):
	"""local backend"""

	def __init__(self, name: Literal['local_simulator']) -> None:
		super().__init__(name)
		if name == 'local_simulator':
			self._backend = LocalEmulator()

	def get_device(self):
		return self._backend

	def get_args(self):
		return {}
