"""Basic problems for Orca"""

import numpy as np
from pyqubo import Array

import qlauncher.problems.problem_initialization as problem
from qlauncher.base import adapter, formatter
# from qlauncher.problems.problem_formulations.jssp.pyqubo_scheduler import get_jss_bqm


# @formatter(problem.JSSP, 'qubo')
# class JSSPOrca:
# 	gamma = 1
# 	lagrange_one_hot = 1
# 	lagrange_precedence = 2
# 	lagrange_share = 5

# 	def _fix_get_jss_bqm(
# 		self, instance, max_time, config, lagrange_one_hot=0, lagrange_precedence=0, lagrange_share=0
# 	) -> tuple[dict, list, None]:
# 		pre_result = get_jss_bqm(
# 			instance,
# 			max_time,
# 			config,
# 			lagrange_one_hot=lagrange_one_hot,
# 			lagrange_precedence=lagrange_precedence,
# 			lagrange_share=lagrange_share,
# 		)
# 		result = (pre_result.spin.linear, pre_result.spin.quadratic, pre_result.spin.offset)  # I need to change it into dict somehow
# 		return result, list(result[0].keys()), None

# 	def calculate_instance_size(self, problem: problem.JSSP):
# 		# Calculate instance size for training
# 		_, variables, _ = self._fix_get_jss_bqm(
# 			problem.instance,
# 			problem.max_time,
# 			self.config,
# 			lagrange_one_hot=self.lagrange_one_hot,
# 			lagrange_precedence=self.lagrange_precedence,
# 			lagrange_share=self.lagrange_share,
# 		)
# 		return len(variables)

# 	def get_len_all_jobs(self, problem: problem.JSSP):
# 		result = 0
# 		for job in problem.instance.values():
# 			result += len(job)
# 		return result

# 	def one_hot_to_jobs(self, binary_vector, problem: problem.JSSP):
# 		actually_its_qubo, variables, model = self._fix_get_jss_bqm(
# 			problem.instance,
# 			problem.max_time,
# 			self.config,
# 			lagrange_one_hot=self.lagrange_one_hot,
# 			lagrange_precedence=self.lagrange_precedence,
# 			lagrange_share=self.lagrange_share,
# 		)
# 		return [variables[i] for i in range(len(variables)) if binary_vector[i] == 1]

# 	def _set_config(self) -> None:
# 		self.config = {}
# 		self.config['parameters'] = {}
# 		self.config['parameters']['job_shop_scheduler'] = {}
# 		self.config['parameters']['job_shop_scheduler']['problem_version'] = 'optimization'

# 	def __call__(self, problem: problem.JSSP):
# 		# Define the matrix Q used for QUBO
# 		self.config = {}
# 		self.instance_size = self.calculate_instance_size(problem)
# 		self._set_config()
# 		actually_its_qubo, variables, model = self._fix_get_jss_bqm(
# 			problem.instance,
# 			problem.max_time,
# 			self.config,
# 			lagrange_one_hot=self.lagrange_one_hot,
# 			lagrange_precedence=self.lagrange_precedence,
# 			lagrange_share=self.lagrange_share,
# 		)
# 		reverse_dict_map = {v: i for i, v in enumerate(variables)}

# 		Q = np.zeros((self.instance_size, self.instance_size))

# 		for (label_i, label_j), value in actually_its_qubo[1].items():
# 			i = reverse_dict_map[label_i]
# 			j = reverse_dict_map[label_j]
# 			Q[i, j] += value
# 			Q[j, i] = Q[i, j]

# 		for label_i, value in actually_its_qubo[0].items():
# 			i = reverse_dict_map[label_i]
# 			Q[i, i] += value
# 		return Q / max(np.max(Q), -np.min(Q)), 0


@formatter(problem.Raw, 'qubo')
def get_raw_qubo(problem: problem.Raw):
	return problem.instance


# @formatter(problem.GraphColoring, 'qubo')
# def get_graph_coloring_qubo(problem: problem.GraphColoring):
# 	"""Returns Qubo function"""
# 	num_qubits = problem.instance.number_of_nodes() * problem.num_colors
# 	x = Array.create('x', shape=(problem.instance.number_of_nodes(), problem.num_colors), vartype='BINARY')
# 	qubo = 0
# 	for node in problem.instance.nodes:
# 		qubo += (1 - sum(x[node, i] for i in range(problem.num_colors))) ** 2
# 	for n1, n2 in problem.instance.edges:
# 		for c in range(problem.num_colors):
# 			qubo += x[n1, c] * x[n2, c]
# 	model = qubo.compile()
# 	qubo_dict, offset = model.to_qubo()
# 	Q_matrix = np.zeros((num_qubits, num_qubits))
# 	for i in range(num_qubits):
# 		for j in range(num_qubits):
# 			n1, c1 = i // problem.num_colors, i % problem.num_colors
# 			n2, c2 = j // problem.num_colors, j % problem.num_colors
# 			key = ('x[' + str(n1) + '][' + str(c1) + ']', 'x[' + str(n2) + '][' + str(c2) + ']')
# 			if key in qubo_dict:
# 				Q_matrix[i, j] = qubo_dict[key]
# 	return Q_matrix, offset


# @formatter(problem=problem.Knapsack, alg_format='qubo')
# def knapsack_qubo(problem: problem.Knapsack, penalty_weight: float = 2.0, value_weight: float = 1.0):
# 	"""
# 	Returns QUBO function for Knapsack problem.
# 	"""
# 	values = problem.values
# 	weights = problem.weights
# 	n = len(values)

# 	x = Array.create('a_x', shape=n, vartype='BINARY')

# 	m = 1 if problem.capacity == 0 else int(np.ceil(np.log2(problem.capacity + 1)))
# 	y = Array.create('z_y', shape=m, vartype='BINARY')
# 	slack = sum((2**k) * y[k] for k in range(m))

# 	weight_sum = sum(weights[i] * x[i] for i in range(n))
# 	if not isinstance(weight_sum, Array):
# 		raise TypeError
# 	penalty = (weight_sum + slack - problem.capacity) ** 2
# 	value_term = sum(values[i] * x[i] for i in range(n))
# 	H = penalty_weight * penalty - value_weight * value_term

# 	qubo_dict, offset = H.compile().to_qubo()
# 	var_labels = [f'z_y[{k}]' for k in range(m)] + [f'a_x[{i}]' for i in reversed(range(n))]
# 	N = len(var_labels)
# 	Q = np.zeros((N, N))
# 	for i, vi in enumerate(var_labels):
# 		for j, vj in enumerate(var_labels):
# 			key = (vi, vj)
# 			if key in qubo_dict:
# 				Q[i, j] = qubo_dict[key]

# 	return Q, float(offset)
