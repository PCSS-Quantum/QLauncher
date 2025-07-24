Examples
========

| Using QLauncher is simple. 
| Simply specify the problem, algorithm and backend you want to use, and QLauncher will take care of the rest.
| Here are some examples to get you started.
| (These examples are also available in the examples directory of the repository.)

----------------
Qiskit
----------------

::

    from qlauncher import *
    from qlauncher.routines.qiskit_routines import QiskitBackend, QAOA

    pr = problems.JSSP(3, 'exact', instance_name='toy', optimization_problem=True)
    alg = QAOA()
    backend = QiskitBackend('local_simulator')

    launcher = QLauncher(pr, alg, backend)
    
    result = launcher.run()
    print(result)



----------------
Dwave
----------------

::

    from qlauncher import *
    from qlauncher.routines.dwave_routines import SimulatedAnnealingBackend, DwaveSolver

    problem = problems.MaxCut(instance_name='default')
    alg = DwaveSolver(1)
    backend = SimulatedAnnealingBackend('local')

    launcher = QLauncher(problem, alg, backend)
    
    result = launcher.run()
    print(result)


----------------
Orca
----------------

::

    from qlauncher import *
    from qlauncher.routines.orca_routines import OrcaBackend, BBS

    problem = problems.MaxCut(instance_name='default')
    alg = BBS()
    backend = OrcaBackend('local')

    launcher = QLauncher(problem, alg, backend)

    result = launcher.run()
    print(result)

