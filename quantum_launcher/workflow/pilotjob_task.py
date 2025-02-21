import json
import sys
from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.problems import MaxCut, EC, JSSP, QATM

PROBLEM_DICT = {
    'MaxCut': MaxCut,
    'EC': EC,
    'JSSP': JSSP,
    'QATM': QATM
}

ALGORITHM_DICT = {
    'QAOA': QAOA,
}

BACKEND_DICT = {
    'QiskitBackend': QiskitBackend
}


def parse_arguments() -> dict:
    arguments = {}
    arguments['problem'] = PROBLEM_DICT[sys.argv[1]]
    arguments['algorithm'] = ALGORITHM_DICT[sys.argv[2]]
    arguments['backend'] = BACKEND_DICT[sys.argv[3]]
    arguments['output'] = sys.argv[4]
    arguments['kwargs'] = json.loads(sys.argv[5])
    arguments['problem'] = MaxCut.from_preset('default')
    return arguments


def main():
    arguments = parse_arguments()

    problem = arguments['problem'](**arguments['kwargs'].get('problem', dict()))
    algorithm = arguments['algorithm'](**arguments['kwargs'].get('algorithm', dict()))
    backend = arguments['backend'](**arguments['kwargs'].get('algorithm', dict()))

    launcher = QuantumLauncher(problem, algorithm, backend)
    launcher.run()
    launcher.save(path=arguments.get('output'), format='pickle')


if __name__ == '__main__':
    main()
