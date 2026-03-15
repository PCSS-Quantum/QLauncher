"""Utility functions for QLauncher"""

from collections import defaultdict
from itertools import chain
from typing import TypeVar

import numpy as np

T = TypeVar('T')


def sum_counts(*counts: dict[T, int]) -> dict[T, int]:
    """Sum up counts from multiple count dicts into one count dict"""
    result = defaultdict(int)
    for key, value in chain(*(c.items() for c in counts)):
        result[key] += value
    return result


def int_to_bitstring(number: int, total_bits: int):
    return np.binary_repr(number, total_bits)[::-1]
