""" Utility functions for QLauncher"""
import numpy as np


def int_to_bitstring(number: int, total_bits: int):
    return np.binary_repr(number, total_bits)[::-1]
