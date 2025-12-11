from qlauncher.problems import EC, JSSP, QATM, TSP, GraphColoring, Knapsack, MaxCut
from qlauncher.routines.qiskit.mitigation_suppression.mitigation import PauliTwirling, NoMitigation, ZeroNoiseExtrapolation

PROBLEM_MAP = {
	'ec': EC.from_preset('micro'),
	'maxcut': MaxCut.from_preset('default'),
	'qatm': QATM.from_preset('rcp-3'),
	'tsp': TSP.generate_tsp_instance(3),
	'gc': GraphColoring.from_preset('small'),
	'knapsack': Knapsack.from_preset('small'),
	'JSSP': JSSP.from_preset('default'),
}

ALL_PROBLEMS = list(PROBLEM_MAP.keys())

MITIGATION_MAP = {
	'zne': ZeroNoiseExtrapolation(),
	'pauli-twirl': PauliTwirling(4),
	'none': NoMitigation(),
}

ALL_MITIGATION_STRATEGIES = list(MITIGATION_MAP.keys())
