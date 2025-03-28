Quantum Launcher Docs
=====================

----------------
About project
----------------
Quantum Launcher is a high-level python library that aims to simplify usage of different quantum algorithms. The goal is to make learning, using and benchmarking different quantum algorithms, hardware and problem formulations simpler.

----------------
Main idea
----------------
Quantum Launcher splits solving problems on Quantum Machine into 3 main components:

- Problem: Formulation of the problem that we want to solve, for example: Maxcut or Exact Cover
- Algorithm: Algorithm implementation that we want to use for solving problem, for example: QAOA, FALQON, BBS
- Backend: The Hardware or local simulator that we want to use to execute our algorithm

----------------
Installation
----------------
You can install quantum launcher using pip

::

   pip install git+https://github.com/psnc-qcg/QCG-QuantumLauncher@QL-2.0

----------------
Optional install
----------------
Quantum Launcher aims to work for many different architectures. Therefore in order to remain compatible with all of them Quantum Launcher by default installs only necessary requirements allowing user to decide what frameworks does one want to use. To make installation easier, there is a bunch of downloads that can be done with optional dependencies, for example:

::

   pip install "git+https://github.com/psnc-qcg/QCG-QuantumLauncher@QL-2.0[qiskit]"

* qiskit: support for IBM's qiskit algorithms and backends.
* orca: support for Orca Computing algorithms and backends NOTE library ptseries is not public therefore one needs to install it on its own.
* dwave: support for D-Wave Systems algorithms and backends.
* cirq: support for Google's cirq backends.
* pilotjob: support for advanced job scheduling using Quantum Launcher and QCG PilotJob for more complex algorithm.


Then you can get to know the library by checking out the following sections:

.. toctree::
   :maxdepth: 1

   examples
   tutorials
   api_reference
