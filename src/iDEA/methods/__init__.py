from enum import Enum


class Methods(Enum):
    INTERACTING = 1
    NON_INTERACTING = 2
    HARTREE = 3
    HARTREE_FOCK = 4
    LDA = 5
    HYBRID = 6