"""Contains many utilities needed for efficient iDEA usage."""


import pickle
import numpy as np


__all__ = ['Container', 'Experiment', 'save_experiment', 'load_experiment']


class Container:
    """Empty container."""
    pass


class Experiment(Container):
    """Container to hold all results, quantities and definitions for an experiment."""
    pass


def save_experiment(experiment: Experiment, file_name: str) -> None:
    """Save an experiment to an experiment file."""
    pickle.dump(experiment, open(file_name, "wb"))


def load_experiment(file_name: str) -> Experiment:
    """Load an experiment from an experiment file."""
    return pickle.load(open(file_name, "rb"))
