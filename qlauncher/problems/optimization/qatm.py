from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from qlauncher import hampy, models
from qlauncher.base.base import Problem
from qlauncher.hampy.object import Equation
from qlauncher.problems.optimization.ec import ring_ham


class QATM(Problem):
	def __init__(
		self,
		cm: np.ndarray,
		aircrafts: pd.DataFrame,
		onehot: Literal['exact', 'quadratic', 'xor'] = 'exact',
		optimization: bool = True,
	) -> None:
		self.cm = cm
		self.aircrafts = aircrafts
		self.onehot = onehot
		self.optimization = optimization

		self.size = len(self.cm)

	@staticmethod
	def from_preset(instance_name: Literal['rcp-3'], **kwargs) -> 'QATM':
		match instance_name:
			case 'rcp-3':
				cm = np.array(
					[
						[1, 0, 1, 0, 0, 0],
						[0, 1, 0, 0, 0, 1],
						[1, 0, 1, 0, 1, 0],
						[0, 0, 0, 1, 0, 0],
						[0, 0, 1, 0, 1, 0],
						[0, 1, 0, 0, 0, 1],
					]
				)
				aircrafts = pd.DataFrame(
					{
						'manouver': ['A0', 'A1', 'A2', 'A0_a=10', 'A1_a=10', 'A2_a=10'],
						'aircraft': ['A0', 'A1', 'A2', 'A0', 'A1', 'A2'],
					}
				)
			case _:
				raise KeyError
		return QATM(cm, aircrafts)

	@classmethod
	def from_file(cls, path: str, instance_name: str = 'QATM', optimization: bool = True) -> 'QATM':
		cm_path = Path(path, 'CM_' + instance_name)
		aircrafts_path = Path(path, 'aircrafts_' + instance_name)

		return QATM(
			np.loadtxt(cm_path),
			pd.read_csv(aircrafts_path, delimiter=' ', names=['manouver', 'aircraft']),
			'exact',
			optimization,
		)

	def to_hamiltonian(self, onehot: Literal['exact', 'quadratic', 'xor'] = 'exact') -> models.Hamiltonian:
		cm = self.cm
		aircrafts = self.aircrafts
		size = len(cm)

		onehot_hamiltonian = Equation(size)
		for _, manouvers in aircrafts.groupby(by='aircraft'):
			if onehot == 'exact':
				h = ~hampy.one_in_n(manouvers.index.values.tolist(), size)
			elif onehot == 'quadratic':
				h = hampy.one_in_n(manouvers.index.values.tolist(), size, quadratic=True)
			elif onehot == 'xor':
				total = Equation(size)
				for part in manouvers.index.values.tolist():
					total ^= total[part]
				h = (~total).hamiltonian

			onehot_hamiltonian += h

		triu = np.triu(cm, k=1)
		conflict_hamiltonian = Equation(size)
		for p1, p2 in zip(*np.where(triu == 1), strict=True):
			eq = Equation(size)
			conflict_hamiltonian += (eq[int(p1)] & eq[int(p2)]).hamiltonian

		hamiltonian = onehot_hamiltonian + conflict_hamiltonian

		if self.optimization:
			goal_hamiltonian = Equation(size)
			for i, (maneuver, ac) in self.aircrafts.iterrows():
				if not isinstance(i, int):
					raise TypeError
				if maneuver != ac:
					goal_hamiltonian += goal_hamiltonian.get_variable(i)
			hamiltonian += goal_hamiltonian / cm.sum().sum()

		return models.Hamiltonian(
			hamiltonian,
			mixer_hamiltonian=self.get_mixer_hamiltonian(),
			initial_state=self.get_initial_state(),
		)

	def get_mixer_hamiltonian(self) -> Equation:
		mixer_hamiltonian = Equation(self.size)
		for _, manouvers in self.aircrafts.groupby(by='aircraft'):
			h = ring_ham(manouvers.index.values.tolist(), self.size)
			mixer_hamiltonian += h
		return mixer_hamiltonian

	def get_initial_state(self) -> QuantumCircuit:
		qc = QuantumCircuit(self.size)
		for _, manouvers in self.aircrafts.groupby(by='aircraft'):
			qc.x(manouvers.index.values.tolist()[0])
		return qc

	def analyze_result(self, result: dict) -> dict[str, np.ndarray]:
		"""
		Analyzes the result in terms of collisions and violations of onehot constraint.

		Parameters:
			result (dict): A dictionary where keys are bitstrings and values are probabilities.

		Returns:
			dict: A dictionary containing collisions, onehot violations, and changes as ndarrays.
		"""
		keys = list(result.keys())
		vectorized_result = np.fromstring(' '.join(list(''.join(keys))), 'u1', sep=' ').reshape(len(result), -1)
		cm = self.cm.copy().astype(int)
		np.fill_diagonal(cm, 0)
		collisions = np.einsum('ij,ij->i', vectorized_result @ cm, vectorized_result) / 2

		df = pd.DataFrame(vectorized_result.transpose())
		df['aircraft'] = self.aircrafts['aircraft']
		onehot_violations = (df.groupby(by='aircraft').sum() != 1).sum(axis=0).to_numpy()

		df['manouver'] = self.aircrafts['manouver']
		no_changes = df[df['aircraft'] == df['manouver']]
		changes = (len(no_changes) - no_changes.drop(['manouver', 'aircraft'], axis=1).sum()).to_numpy().astype(int)
		changes[onehot_violations != 0] = -1

		at_least_one = (df.loc[:, df.columns != 'manouver'].groupby('aircraft').sum() > 0).all().to_numpy().astype(int)

		return {'collisions': collisions, 'onehot_violations': onehot_violations, 'changes': changes, 'at_least_one': at_least_one}
