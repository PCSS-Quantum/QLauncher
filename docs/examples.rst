Examples
========

| Using Quantum Launcher is simple. 
| Simply specify the problem, algorithm and backend you want to use, and Quantum Launcher will take care of the rest.
| Here are some examples to get you started.
| (These examples are also available in the examples directory of the repository.)

----------------
Qiskit
----------------

::

    from quantum_launcher import *
    from quantum_launcher.routines.qiskit_routines import QiskitBackend, QAOA

    pr = problems.JSSP(3, 'exact', instance_name='toy', optimization_problem=True)
    alg = QAOA()
    backend = QiskitBackend('local_simulator')

    launcher = QuantumLauncher(pr, alg, backend)
    
    result = launcher.run()
    print(result)



----------------
Dwave
----------------

::

    from quantum_launcher import *
    from quantum_launcher.routines.dwave_routines import SimulatedAnnealingBackend, DwaveSolver

    problem = problems.MaxCut(instance_name='default')
    alg = DwaveSolver(1)
    backend = SimulatedAnnealingBackend('local')

    launcher = QuantumLauncher(problem, alg, backend)
    
    result = launcher.run()
    print(result)


----------------
Orca
----------------

::

    from quantum_launcher import *
    from quantum_launcher.routines.orca_routines import OrcaBackend, BBS

    problem = problems.MaxCut(instance_name='default')
    alg = BBS()
    backend = OrcaBackend('local')

    launcher = QuantumLauncher(problem, alg, backend)

    result = launcher.run()
    print(result)

