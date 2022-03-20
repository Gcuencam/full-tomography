import getopt
import os
import sys

from src.lib.measurement_collector import collect_pauli_measurements, collect_povm_measurements

os.environ['DEBUG'] = 'False'


def main(argv):
    shots = 0
    qc_size = 0
    output_file_name = 'measurements.txt'
    measurement_type = ''
    try:
        opts, args = getopt.getopt(argv, "dhs:q:o:m:", ["shots=", "qubits=", "output=", "measurement="])
    except getopt.GetoptError:
        print('test.py -s <shoots_number> -q <qubits_number> -o <filename>-d')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print()
            sys.exit()
        elif opt in ("-s", "--shoots"):
            shots = int(arg)
        elif opt in ("-q", "--qubits"):
            qc_size = int(arg)
        elif opt in ("-d", "--debug"):
            os.environ['DEBUG'] = 'True'
        elif opt in ("-o", "--output"):
            output_file_name = arg
        elif opt in ("-m", "--measurement"):
            measurement_type = arg

    if measurement_type == 'povm':
        collect_povm_measurements(qc_size, shots, output_file_name)
    elif measurement_type == 'pauli':
        collect_pauli_measurements(qc_size, shots, output_file_name)
    else:
        print("wrong measurement type: valid options [povm | pauli]")


if __name__ == '__main__':
    main(sys.argv[1:])
