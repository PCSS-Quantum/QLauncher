from qlauncher.problems import EC, JSSP, QATM, TSP, GraphColoring, Knapsack, MaxCut

PROBLEM_MAP = {
	'ec': EC.from_preset('micro'),
	'maxcut': MaxCut.from_preset('default'),
	'qatm': QATM.from_preset('rcp-3'),
}

ALL_PROBLEMS = list(PROBLEM_MAP.keys())
