from qlauncher import QLauncher
from qlauncher.problems import MaxCut
from qlauncher.routines.qiskit import IBMBackend
from qlauncher.routines.qiskit.algorithms import EducatedGuess

pr = MaxCut.from_preset('default')
educated_guess = EducatedGuess(starting_p=3, max_p=8, verbose=True)
backend = IBMBackend('local_simulator')
launcher = QLauncher(pr, educated_guess, backend)

inform = launcher.run()
