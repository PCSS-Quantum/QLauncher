from qlauncher.problems import EC, JSSP, QATM, TSP, GraphColoring, Knapsack, MaxCut

PROBLEM_MAP = {
	'ec': EC.from_preset('micro'),
	'maxcut': MaxCut.from_preset('default'),
	'qatm': QATM.from_preset('rcp-3'),
	'tsp': TSP.generate_tsp_instance(3),
	'gc': GraphColoring.from_preset('small'),
	'knapsack': Knapsack.from_preset('small'),
	'JSSP':JSSP.from_preset('default')
}

ALL_PROBLEMS = list(PROBLEM_MAP.keys())
