from . import w, ghz, plus
from enum import Enum


class States(Enum):
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
