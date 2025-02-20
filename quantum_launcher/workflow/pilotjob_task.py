import json
import sys
from quantum_launcher import QuantumLauncher
from quantum_launcher.base import Result
from quantum_launcher.routines.qiskit_routines import QAOA, QiskitBackend
from quantum_launcher.problems import MaxCut, EC, JSSP, QATM

PROBLEM_DICT = {
    'MaxCut': MaxCut,
    'EC': EC,
    'JSSP': JSSP,
    'QATM': QATM
}


def parse_arguments() -> dict:
    arguments = {}
    arguments['output'] = sys.argv[1]
    arguments['kwargs'] = json.loads(sys.argv[2])
    arguments['problem'] = MaxCut.from_preset('default')
    return arguments


def check_cores() -> int: ...


def save_result(result: Result, path: str):
    with open(path, 'w') as f:
        json.dump(result.__dict__, f)


def main():
    arguments = parse_arguments()

    problem = arguments['problem']
    algorithm = QAOA(**arguments['kwargs'].get('algorithm', dict()))
    backend = QiskitBackend('local_simulator')

    launcher = QuantumLauncher(problem, algorithm, backend)
    launcher.run()
    launcher.save(path=arguments.get('output'), format='pickle')


if __name__ == '__main__':
    main()
