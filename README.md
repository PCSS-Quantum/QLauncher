<h1 style="text-align:center;background:linear-gradient(rgb(141, 222, 251),rgb(0, 21, 255));  -webkit-background-clip: text;-webkit-text-fill-color: transparent;"> Quantum Launcher </h1>

<h2 style="text-align:center;"> About Project </h2>

<p style="text-align:center;">
Quantum Launcher is a high-level python library that aims to simplify usage of different quantum algorithms. The goal is to make learning, using and benchmarking different quantum algorithms, hardware and problem formulations simpler.
</p>

<h2 style="text-align:center;"> Main Idea </h2>
Quantum Launcher splits solving problems on Quantum Machine into 3 main components:

- Problem: Formulation of the problem that we want to solve, for example: Maxcut or Exact Cover
- Algorithm: Algorithm implementation that we want to use for solving problem, for example: QAOA, FALQON, BBS
- Backend: The Hardware or local simulator that we want to use to execute our algorithm

![Quantum Launcher](.figures/QCG-QL.png)

<h2 style="text-align:center;"> Supported Features </h2>

So far Quantum Launcher provides user with:

- High-level architecture for executing problems
- Set of predefined problems, algorithms, and backends
- Automated processing of the problem
- Asynchronous architecture to execute problems either standalone or in a grid

Features planned to be implemented in feature:
