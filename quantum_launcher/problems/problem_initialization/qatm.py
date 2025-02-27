"""  This module contains the QATM class."""
import os

import numpy as np
import pandas as pd

from quantum_launcher.base import Problem


class QATM(Problem):
    """ 
    Class for Quantum Aircraft Traffic Management (QATM) problem.

    This class represents Quantum Aircraft Traffic Management (QATM) problem which is a combinatorial optimization problem
    that involves scheduling a set of aircrafts on a set of manouvers. Each aircraft consists of a sequence of manouvers
    that must be performed. The objective is to find a schedule that minimizes the number of collisions between aircrafts.
    The class contains an instance of the problem, so it can be passed into Quantum Launcher.

    Attributes:
        onehot (str): The one-hot encoding used for the problem.
        instance (dict): The instance of the problem.
        instance_name (str | None): The name of the instance.
        instance_path (str): The path to the instance file.

    """

    def __init__(self, onehot: str = 'exact', instance: any = None, instance_name: str | None = None,
                 optimization_problem: bool = False) -> None:
        super().__init__(instance=instance, instance_name=instance_name)
        self.onehot = onehot
        self.optimization_problem = optimization_problem

    @property
    def setup(self) -> dict:
        return {
            'onehot': self.onehot,
            'instance_name': self.instance_name
        }

    def _get_path(self) -> str:
        return f'{self.name}@{self.instance_name.split(".", 1)[0]}'

    @classmethod
    def from_file(cls, instance_path: str, instance_name: str):
        cm_path = os.path.join(instance_path, 'CM_' + instance_name)
        aircrafts_path = os.path.join(
            instance_path, 'aircrafts_' + instance_name)

        instance = {'cm': np.loadtxt(cm_path), 'aircrafts': pd.read_csv(aircrafts_path, delimiter=' ', names=['manouver', 'aircraft'])}
        return QATM('exact', instance, instance_name)

    def analyze_result(self, result: dict):
        """
        Analyzes the result in terms of collisions and violations of onehot constraint.

        Parameters:
            result (dict): A dictionary where keys are bitstrings and values are probabilities.

        Returns:
            dict: A dictionary containing collisions, onehot violations, and changes as ndarrays.
        """
        keys = list(result.keys())
        vectorized_result = (np.fromstring(
            "".join(keys), 'u1') - ord('0')).reshape(len(result), -1)

        cm = self.instance['cm'].copy().astype(int)
        np.fill_diagonal(cm, 0)
        collisions = np.einsum(
            'ij,ij->i', vectorized_result @ cm, vectorized_result) / 2

        df = pd.DataFrame(vectorized_result.transpose())
        df['aircraft'] = self.instance['aircrafts']['aircraft']
        onehot_violations = (df.groupby(by='aircraft').sum()
                             != 1).sum(axis=0).ravel()

        df['manouver'] = self.instance['aircrafts']['manouver']
        no_changes = df[df['aircraft'] == df['manouver']]
        changes = (len(no_changes) - no_changes.drop(
            ['manouver', 'aircraft'], axis=1).sum()).ravel().astype(int)
        changes[onehot_violations != 0] = -1

        at_least_one = (df.loc[:, df.columns != 'manouver'].groupby(
            'aircraft').sum() > 0).all().ravel().astype(int)

        return {'collisions': collisions,
                'onehot_violations': onehot_violations,
                'changes': changes,
                'at_least_one': at_least_one}
