from qlauncher.problems import EC, JSSP, QATM, TSP, GraphColoring, Knapsack, MaxCut, VertexCover
from qlauncher.routines.qiskit.mitigation_suppression.mitigation import (
	NoMitigation,
	PauliTwirling,
	WeighedMitigation,
	ZeroNoiseExtrapolation,
)

PROBLEM_MAP = {
	'ec': EC.from_preset('micro'),
	'maxcut': MaxCut.from_preset('default'),
	'qatm': QATM.from_preset('rcp-3'),
	'tsp': TSP.generate_tsp_instance(3),
	'gc': GraphColoring.from_preset('small'),
	'knapsack': Knapsack.from_preset('small'),
	'JSSP': JSSP.from_preset('default'),
	'VertexCover': VertexCover.from_preset('default'),
}

ALL_PROBLEMS = list(PROBLEM_MAP.keys())

MITIGATION_MAP = {
	'zne': ZeroNoiseExtrapolation(),
	'pauli-twirl': PauliTwirling(4),
	'none': NoMitigation(),
	'weighed-equal': WeighedMitigation([NoMitigation(), PauliTwirling(2)]),
	'weighed-mixed': WeighedMitigation([NoMitigation(), PauliTwirling(2)], method_weights=[0.5, 1.5]),
}

ALL_MITIGATION_STRATEGIES = list(MITIGATION_MAP.keys())
