import getopt
import os
import sys

from src.measurement_collector import collect_sic_povm_measurements, collect_pauli_measurements
from src.states.builder import States

os.environ['DEBUG'] = 'False'


def main(argv):
    shots = 0
    qc_size = 0
    output_file_name = 'measurements.txt'
    measurement_type = ''
    try:
        opts, args = getopt.getopt(argv, "dhs:q:o:m:t:", ["shots=", "qubits=", "output=", "measurement=", "type="])
    except getopt.GetoptError:
        print('test.py -s <shots_number> -q <qubits_number> -o <filename> -d -t type')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print()
            sys.exit()
        elif opt in ("-s", "--shots"):
            shots = int(arg)
        elif opt in ("-q", "--qubits"):
            qc_size = int(arg)
        elif opt in ("-d", "--debug"):
            os.environ['DEBUG'] = 'True'
        elif opt in ("-o", "--output"):
            output_file_name = arg
        elif opt in ("-m", "--measurement"):
            measurement_type = arg
        elif opt in ("-t", "--type"):
            state_type = States(arg)

    if measurement_type == 'sic':
        collect_sic_povm_measurements(state_type, qc_size, shots, output_file_name)
    elif measurement_type == 'pauli':
        collect_pauli_measurements(state_type, qc_size, shots, output_file_name)
    else:
        print("wrong measurement type: valid options [sic | pauli]")


if __name__ == '__main__':
    main(sys.argv[1:])
