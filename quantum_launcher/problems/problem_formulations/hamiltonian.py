""" Hamiltonian formulation of problems """
from itertools import product
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_ising

from quantum_launcher.base import formatter
from quantum_launcher.base.adapter_structure import adapter
import quantum_launcher.problems.problem_initialization as problems
import quantum_launcher.hampy as hampy
from quantum_launcher.hampy import Equation, Variable
from quantum_launcher.problems.problem_formulations.hamiltonians.tsp import problem_to_hamiltonian as tsp_to_hamiltonian


@adapter('hamiltonian', 'qubo', onehot='quadratic')
def hamiltonian_to_qubo(hamiltonian):
    qp = from_ising(hamiltonian)
    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp).objective
    return qubo.quadratic.to_array(), 0


@formatter(problems.EC, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.EC, onehot='exact'):
    """ generating hamiltonian"""
    elements = set().union(*problem.instance)
    onehots = []
    for ele in elements:
        ohs = set()
        for i, subset in enumerate(problem.instance):
            if ele in subset:
                ohs.add(i)
        onehots.append(ohs)
    hamiltonian = None
    for ohs in onehots:
        if onehot == 'exact':
            part = (~hampy.one_in_n(list(ohs), len(problem.instance))).hamiltonian
        elif onehot == 'quadratic':
            part = hampy.one_in_n(list(ohs), len(problem.instance), quadratic=True).hamiltonian
        if hamiltonian is None:
            hamiltonian = part
        else:
            hamiltonian += part
    return hamiltonian.simplify()


@formatter(problems.JSSP, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.JSSP) -> SparsePauliOp:
    if problem.optimization_problem:
        return problem.h_o
    else:
        return problem.h_d


@formatter(problems.MaxCut, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.MaxCut):
    ham = None
    n = problem.instance.number_of_nodes()
    for edge in problem.instance.edges():
        if ham is None:
            ham = ~hampy.one_in_n(edge, n)
        else:
            ham += ~hampy.one_in_n(edge, n)
    return ham.hamiltonian.simplify()


@formatter(problems.QATM, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.QATM, onehot='exact'):
    cm = problem.instance['cm']
    aircrafts = problem.instance['aircrafts']

    onehot_hamiltonian = None
    for plane, manouvers in aircrafts.groupby(by='aircraft'):
        if onehot == 'exact':
            h = (~hampy.one_in_n(manouvers.index.values.tolist(), len(cm))).hamiltonian
        elif onehot == 'quadratic':
            h = hampy.one_in_n(manouvers.index.values.tolist(), len(cm), quadratic=True).hamiltonian
        elif onehot == 'xor':
            total = None
            eq = Equation(len(cm))
            for part in manouvers.index.values.tolist():
                if total is None:
                    total = eq[part].to_equation()
                    continue
                total ^= eq[part]
            h = (~total).hamiltonian
        if onehot_hamiltonian is not None:
            onehot_hamiltonian += h
        else:
            onehot_hamiltonian = h

    triu = np.triu(cm, k=1)
    conflict_hamiltonian = None
    for p1, p2 in zip(*np.where(triu == 1)):
        eq = Equation(len(cm))
        partial_hamiltonian = (eq[int(p1)] & eq[int(p2)]).hamiltonian
        if conflict_hamiltonian is not None:
            conflict_hamiltonian += partial_hamiltonian
        else:
            conflict_hamiltonian = partial_hamiltonian

    hamiltonian = onehot_hamiltonian + conflict_hamiltonian

    if problem.optimization_problem:
        goal_hamiltonian = None
        for i, (maneuver, ac) in problem.instance['aircrafts'].iterrows():
            if maneuver != ac:
                eq = Equation(len(aircrafts))
                h = Variable(i, eq).to_equation()
                if goal_hamiltonian is None:
                    goal_hamiltonian = h
                else:
                    goal_hamiltonian += h
        goal_hamiltonian /= sum(sum(cm))
        hamiltonian += goal_hamiltonian

    return hamiltonian.simplify()


@formatter(problems.Raw, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.Raw) -> SparsePauliOp:
    return problem.instance


@formatter(problems.TSP, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.TSP, onehot='exact', constraints_weight=1, costs_weight=1) -> SparsePauliOp:
    return tsp_to_hamiltonian(
        problem,
        onehot=onehot,
        constraints_weight=constraints_weight,
        costs_weight=costs_weight
    )


@formatter(problems.GraphColoring, 'hamiltonian')
def get_qiskit_hamiltonian(problem: problems.GraphColoring, constraints_weight=1, costs_weight=1):
    color_bit_length = int(np.ceil(np.log2(problem.num_colors)))
    num_qubits = problem.instance.number_of_nodes() * color_bit_length
    eq = Equation(num_qubits)
    # Penalty for assigning the same colors to neighboring vertices
    for node1, node2 in problem.instance.edges:
        for ind, comb in enumerate(product(range(2), repeat=color_bit_length)):
            if ind >= problem.num_colors:
                break
            eq2 = None
            for i in range(color_bit_length):
                qubit1 = eq[node1 * color_bit_length + i]
                qubit2 = eq[node2 * color_bit_length + i]
                if comb[i]:
                    exp = qubit1 & qubit2
                else:
                    exp = ~qubit1 & ~qubit2
                if eq2 is None:
                    eq2 = exp
                else:
                    eq2 &= exp
            eq += eq2
    eq *= costs_weight
    # Penalty for using excessive colors
    for node in problem.instance.nodes:
        for ind, comb in enumerate(product(range(2), repeat=color_bit_length)):
            if ind < problem.num_colors:
                continue
            eq2 = None
            for i in range(color_bit_length):
                qubit = eq[node * color_bit_length + i]
                exp = qubit if comb[i] else ~qubit
                if eq2 is None:
                    eq2 = exp
                else:
                    eq2 &= exp
            eq += eq2
    eq *= constraints_weight
    return eq.hamiltonian
