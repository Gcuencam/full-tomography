# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 09:30:04 2022

@author: xelivy
"""

# Create the X-measurement function:
def x_measurement(qc, qubit, cbit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.h(qubit)
    qc.measure(qubit, cbit)
    return qc

# Create the Y-measurement function:
def y_measurement(qc, qubit, cbit):
    """Measure 'qubit' in the X-basis, and store the result in 'cbit'"""
    qc.sdg(qubit)
    qc.h(qubit)
    qc.measure(qubit, cbit)
    return qc