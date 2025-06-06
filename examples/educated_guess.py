from quantum_launcher import QuantumLauncher
from quantum_launcher.problems import MaxCut
from quantum_launcher.routines.qiskit import QiskitBackend
from quantum_launcher.routines.qiskit.algorithms import EducatedGuess

pr = MaxCut.from_preset('default')
educated_guess = EducatedGuess(starting_p=2, max_p=5, verbose=True)
backend = QiskitBackend('local_simulator')
launcher = QuantumLauncher(pr, educated_guess, backend)

inform = launcher.run()
