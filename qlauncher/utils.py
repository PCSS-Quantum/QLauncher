""" Utility functions for QLauncher"""
import numpy as np
from typing import TypeVar
from itertools import chain

T = TypeVar("T")


def sum_counts(*counts: dict[T, int]) -> dict[T, int]:
    """Sum up counts from multiple count dicts into one count dict"""
    result = {}
    for key, value in chain(*(c.items() for c in counts)):
        result[key] = result.get(key, 0) + value
    return result


def int_to_bitstring(number: int, total_bits: int):
    return np.binary_repr(number, total_bits)[::-1]
