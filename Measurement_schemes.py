import random as rand 
import numpy as np
import itertools

#def randomized_localPauli(num_total_measurements, qubit_num):
def randomized_localPauli(qubit_num):
    #    num_total_measurements: int for the total number of measurement rounds
    #    qubit_num: int for how many qubits in the quantum system/circuit
    
    # measurement_list = []
    # for i in range(int(num_total_measurements)):
        #measure_round = [rand.choice(['X','Y','Z']) for j in range(int(qubit_num))]
    #    measure_round = np.array(np.meshgrid(['X'], ['Y'], ['Z'])).T.reshape(-1, 3)
    # print(measure_round)
    # measurement_list.append(measure_round)
    r = itertools.product(['X', 'Y', 'Z'], repeat=qubit_num)
    arr = np.array((list(r)))
    #return measurement_list
    return arr