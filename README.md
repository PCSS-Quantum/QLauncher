# QLauncher

## About Project

QLauncher is a high-level Python library that simplifies the process of running quantum algorithms. The library aims to make it easier to run, test, benchmark, and optimize quantum algorithms by providing tools that work across diverse configurations.

The library contains a rich collection of preset problems and algorithms, eliminating the need to repeatedly implement foundational components such as problem-specific QUBO formulations or Hamiltonians. This approach significantly reduces the overhead when benchmarking different quantum approaches.

QLauncher introduces an intuitive architectural framework by dividing the quantum computation pipeline into three distinct components: Problem, Algorithm, and Backend. This separation creates a universal interface that allows researchers and developers to focus on specific aspects of quantum computation while maintaining compatibility across the entire ecosystem.

![QLauncher](.figures/QL.png)

## Supported features

Additionally to ability of quickly changing tested problem, algorithm or backend QLauncher comes with a bunch of useful features such as:

-   Random problem instances generator.
-   Automatic translation between problem formulations (e.g. QUBO -> Hamiltonian).
-   QASM-based translation to match different frameworks (such as running qiskit's algorithm on cirq's computer).
-   Asynchronous architecture to execute problems either standalone or in a grid.
-   Access to more advanced workflows with qcg-pilotjob.
-   Interface for simple profiling of algorithms.
-   Creation of more complex workflows using WorkflowManager enabling splitting algorithms across multiple devices.

## Installation

To install the following library use the following script:

```sh
pip install qlauncher
```

### Optional Installs

QLauncher aims to work for many different architectures. Therefore in order to remain compatible with all of them QLauncher by default installs only necessary requirements allowing user to decide what frameworks does one want to use. To make installation easier, there is a bunch of downloads that can be done with optional dependencies, for example:

```sh
pip install 'qlauncher[orca]'
```

to install all requirements necessary to run qiskit algorithms.

-   **qiskit**: support for IBM's qiskit algorithms and backends.
-   **orca**: support for Orca Computing algorithms and backends **NOTE** library ptseries is not public therefore one needs to install it on it's own.
-   **dwave**: support for D-Wave Systems algorithms and backends.
-   **cirq**: support for Google's cirq backends.
-   **pilotjob**: support for advanced job scheduling using QLauncher and QCG PilotJob for more complex algorithm.

## Supported problems, algorithms and backends

QLauncher was made to simplify using of multiple different problems, algorithms and backends, therefore adding new things is relatively easy.

Supported problems:

-   MaxCut
-   Exact Cover
-   Job Shop Shedueling
-   Air Traffic Management
-   Traveling Salesman Problem
-   Graph Coloring

For now supported backends are:

-   Qiskit
-   Orca Computing
-   D-wave
-   AQT
-   Cirq

## Usage examples

Main idea of the project was to give a user quick and high level access to many different problems, algorithms and backends keeping interface simple.
For example to solve MaxCut problem with QAOA on qiskit simulator all you need to type is:

```py
# Necessary imports
from qlauncher import QLauncher
from qlauncher.problems import MaxCut
from qlauncher.routines.qiskit import QiskitBackend, QAOA

# Selecting problem, algorithm and backend
problem = MaxCut.from_preset('default')
algorithm = QAOA(p=3)
backend = QiskitBackend('local_simulator')

# Selecting launcher (QLauncher by default, but other can be used for profiling/parallel processing)
launcher = QLauncher(problem, algorithm, backend)

# Running the algorithm
result = launcher.run()
```

A key advantage of our library is that changing algorithms (such as switching to Quantum Annealing from Dwave) doesn't require manually specifying that MaxCut needs to provide QUBO representation - this translation happens automatically behind the scenes.

```py
# Necessary imports
from qlauncher import QLauncher
from qlauncher.problems import MaxCut
from qlauncher.routines.dwave import SimulatedAnnealingBackend, DwaveSolver

# Selecting problem, algorithm and backend
problem = MaxCut.from_preset('default')
algorithm = DwaveSolver()
backend = SimulatedAnnealingBackend('local_simulator')

# Selecting launcher (QLauncher by default, but other can be used for profiling/parallel processing)
launcher = QLauncher(problem, algorithm, backend)

# Running the algorithm
result = launcher.run()
```

## License

This project uses the [To Be determined License](LICENSE).
