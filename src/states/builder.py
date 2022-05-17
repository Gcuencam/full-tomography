from . import w, ghz, plus
from enum import Enum


class States(str, Enum):
    W = 'w'
    GHZ = 'ghz'
    Plus = '+'


def build(type, circuit, referencePosition, size):
    if type == States.W:
        return w.build(circuit, referencePosition, size)
    if type == States.GHZ:
        return ghz.build(circuit, referencePosition, size)
    if type == States.Plus:
        return plus.build(circuit, referencePosition, size)


def get_state_vector(type, size):
    if type == States.W:
        return w.get_w_state_vector(size)
    if type == States.GHZ:
        return ghz.get_ghz_state_vector(size)
    if type == States.Plus:
        return plus.get_plus_state_vector(size)
