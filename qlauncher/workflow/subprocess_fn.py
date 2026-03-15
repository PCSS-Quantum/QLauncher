import sys

import dill

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError(f'Expected 2 args (input, output) files, got {len(sys.argv)}')

    in_file, out_file = sys.argv[1:]

    with open(in_file, 'rb') as f:
        fn = dill.load(f)

    if not callable(fn):
        raise ValueError(f'Expected to unpack a callable object, input.pkl contains {type(fn)}')

    result = fn()

    with open(out_file, 'wb') as f:
        dill.dump(result, f)
