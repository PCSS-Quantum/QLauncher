import json
import os
import sys
from typing import Dict, Tuple, Type
from quantum_launcher import QuantumLauncher
from quantum_launcher.routines.qiskit_routines import QAOA, IBMBackend
from quantum_launcher.problems import MaxCut, EC, JSSP, QATM, Problem
import dill

PROBLEM_DICT: Dict[str, Type[Problem]] = {
    'MaxCut': MaxCut,
    'EC': EC,
    'JSSP': JSSP,
    'QATM': QATM
}

ALGORITHM_DICT = {
    'QAOA': QAOA,
}

BACKEND_DICT = {
    'QiskitBackend': IBMBackend
}


def parse_arguments() -> Tuple[QuantumLauncher, str]:
    """ Returns Quantum Launcher object and output file path """
    if len(sys.argv) == 3:
        input_file_path = sys.argv[1]
        with open(input_file_path, 'rb') as f:
            launcher = dill.load(f)
        os.remove(input_file_path)
        output_path = sys.argv[2]
    elif len(sys.argv) == 6:
        problem = PROBLEM_DICT[sys.argv[1]]
        algorithm = ALGORITHM_DICT[sys.argv[2]]
        backend = BACKEND_DICT[sys.argv[3]]
        kwargs = json.loads(sys.argv[4])
        launcher = QuantumLauncher(problem(**kwargs.get('problem', dict())),
                                   algorithm(**kwargs.get('algorithm', dict())),
                                   backend(**kwargs.get('backend', dict())))
        output_path = sys.argv[5]
    else:
        raise ValueError(f'Wrong number of arguments, expected 3 or 6 got {len(sys.argv)} instead')

    return launcher, output_path


def main():
    launcher, output_path = parse_arguments()

    launcher.run()
    launcher.save(path=output_path, format='pickle')


if __name__ == '__main__':
    main()
