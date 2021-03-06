"""Contains many utilities useful for efficient iDEA usage."""


import pickle


__all__ = [
    "Container",
    "ArrayPlaceholder",
    "Experiment",
    "save_experiment",
    "load_experiment",
]


class Container:
    r"""Empty container."""


class ArrayPlaceholder:
    r"""Array Placeholder."""


class Experiment(Container):
    r"""Container to hold all results, quantities and definitions for an experiment."""


def save_experiment(experiment: Experiment, file_name: str) -> None:
    r"""
    Save an experiment to an experiment file.

    | Args:
    |     experiment: iDEA.utilities.Experiment, Experiment object to save.
    |     file_name: str, file name.
    """
    pickle.dump(experiment, open(file_name, "wb"))


def load_experiment(file_name: str) -> Experiment:
    r"""
    Load an experiment from an experiment file.

    | Args:
    |     file_name: str, file name.

    | Returns
    |     experiment: iDEA.utilities.Experiment, Loaded Experiment object.
    """
    return pickle.load(open(file_name, "rb"))
