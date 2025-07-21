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
Getting started
----------------
To install the quantum launcher use the following command:

::

   pip install quantum-launcher


Then you can get to know the library by checking out the following sections:

.. toctree::
   :maxdepth: 1

   examples
   tutorials
   API Documentation <API/qlauncher>
