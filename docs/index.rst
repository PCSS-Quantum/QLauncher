Quantum Launcher Docs
=====================

----------------
About project
----------------
Quantum Launcher is a high-level Python library that simplifies the process of running quantum algorithms. The library aims to make it easier to run, test, benchmark, and optimize quantum algorithms by providing tools that work across diverse configurations.
The library contains a rich collection of preset problems and algorithms, eliminating the need to repeatedly implement foundational components such as problem-specific QUBO formulations or Hamiltonians. This approach significantly reduces the overhead when benchmarking different quantum approaches.
Quantum Launcher introduces an intuitive architectural framework by dividing the quantum computation pipeline into three distinct components: Problem, Algorithm, and Backend. This separation creates a universal interface that allows researchers and developers to focus on specific aspects of quantum computation while maintaining compatibility across the entire ecosystem.

------------------
Supported features
------------------
Additionally to ability of quickly changing tested problem, algorithm or backend Quantum Launcher comes with a bunch of useful features such as:

* Random problem instances generator.
* Automatic translation between problem formulations (e.g. QUBO -> Hamiltonian).
* QASM-based translation to match different frameworks (such as running qiskit's algorithm on cirq's computer).
* Asynchronous architecture to execute problems either standalone or in a grid.
* Access to more advanced workflows with qcg-pilotjob.
* Interface for simple profiling of algorithms.
* Creation of more complex workflows using WorkflowManager enabling splitting algorithms across multiple devices.
----------------
Installation
----------------
To install the following library use the following script:

::

   pip install quantum-launcher

----------------
Optional install
----------------
Quantum Launcher aims to work for many different architectures. Therefore in order to remain compatible with all of them Quantum Launcher by default installs only necessary requirements allowing user to decide what frameworks does one want to use. To make installation easier, there is a bunch of downloads that can be done with optional dependencies, for example:

::

   pip install 'quantum-launcher[orca]'

to install all requirements necessary to run orca algorithms.

* orca: support for Orca Computing algorithms and backends NOTE library ptseries is not public therefore one needs to install it on its own.
* dwave: support for D-Wave Systems algorithms and backends.
* cirq: support for Google's cirq backends.
* pilotjob: support for advanced job scheduling using Quantum Launcher and QCG PilotJob for more complex algorithm.

----------------
Supported problems, algorithms and backends
----------------
Quantum Launcher was made to simplify using of multiple different problems, algorithms and backends, therefore adding new things is relatively easy.

Supported problems:
* MaxCut
* Exact Cover
* Job Shop Shedueling
* Air Traffic Management
* Traveling Salesman Problem
* Graph Coloring

For now supported backends are:

* Qiskit
* Orca Computing
* D-wave
* AQT
* Cirq

Then you can get to know the library by checking out the following sections:

.. toctree::
   :maxdepth: 1

   examples
   tutorials
   api_reference
