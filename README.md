# Quantum Launcher

## About Project

Quantum Launcher is a high-level python library that aims to simplify usage of different quantum algorithms. The goal is to make learning, using and benchmarking different quantum algorithms, hardware and problem formulations simpler.

<h2 style="text-align:center;"> Main Idea </h2>
Quantum Launcher splits solving problems on Quantum Machine into 3 main components:

- Problem: Formulation of the problem that we want to solve, for example: Maxcut or Exact Cover
- Algorithm: Algorithm implementation that we want to use for solving problem, for example: QAOA, FALQON, BBS
- Backend: The Hardware or local simulator that we want to use to execute our algorithm

![Quantum Launcher](.figures/QL.png)

<h2 style="text-align:center;"> Supported Features </h2>

So far Quantum Launcher provides user with:

- High-level architecture for executing problems
- Set of predefined problems, algorithms, and backends
- Automated processing of the problem
- Asynchronous architecture to execute problems either standalone or in a grid

Features planned to be implemented in feature:

<h2 style="text-align:center;"> Installation and Examples </h2>

## Installation

To install the following library use the following script:

```sh
pip install git+https://github.com/psnc-qcg/QCG-QuantumLauncher@QL-2.0
```

### Optional Installs

Quantum Launcher aims to work for many different architectures. Therefore in order to compatible with all of them Quantum Launcher be default installs only necessary requirements allowing user to decide what frameworks does one want to use. To make installation easier, there is a bunch of downloads that can be done with optional dependencies, for example:

```sh
pip install "git+https://github.com/psnc-qcg/QCG-QuantumLauncher@QL-2.0[qiskit]"
```

to install all requirements necessary to run qiskit algorithms.

## Supported backends

Quantum Launcher was made to simplify using of multiple different backends, therefore adding new backends is relatively easy.

For now supported backends are:

- Qiskit
- Orca Computing
- D-wave
- AQT
- Cirq

## Usage examples

Main idea of the project was to give a user quick and high level access to many different problems, algorithms and backends keeping interface simple.
For example to solve MaxCut problem with QAOA on qiskit simulator all you need to type is:

```py
# Necessary imports
from quantum_launcher import QuantumLauncher
from quantum_launcher.problems import MaxCut
from quantum_launcher.routines.qiskit_routines import QiskitBackend, QAOA

# Selecting problem, algorithm and backend
problem = MaxCut.from_preset('default')
algorithm = QAOA(p=3)
backend = QiskitBackend('local_simulator')

# Selecting launcher (Quantum Launcher by default, but other can be used for profiling/parallel processing)
launcher = QuantumLauncher(problem, algorithm, backend)

# Running the algorithm
result = launcher.run()
```

What the best in our library is that for changing only the algorithm for such as Quantum Annealing from Dwave, you don't actually need to specify that MaxCut will need to give Qubo, as it's done behind the user view.

```py
# Necessary imports
from quantum_launcher import QuantumLauncher
from quantum_launcher.problems import MaxCut
from quantum_launcher.routines.dwave_routines import SimulatedAnnealingBackend, DwaveSolver

# Selecting problem, algorithm and backend
problem = MaxCut.from_preset('default')
algorithm = DwaveSolver()
backend = SimulatedAnnealingBackend('local_simulator')

# Selecting launcher (Quantum Launcher by default, but other can be used for profiling/parallel processing)
launcher = QuantumLauncher(problem, algorithm, backend)

# Running the algorithm
result = launcher.run()
```

## License

This project uses the [To Be determined License](LICENSE).
