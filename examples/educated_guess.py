from qlauncher import QLauncher
from qlauncher.problems import MaxCut
from qlauncher.routines.qiskit_routines import IBMBackend
from qlauncher.routines.qiskit_routines.algorithms import EducatedGuess

pr = MaxCut.from_preset('default')
educated_guess = EducatedGuess(starting_p=2, max_p=5, verbose=True)
backend = IBMBackend('local_simulator')
launcher = QLauncher(pr, educated_guess, backend)

inform = launcher.run()
