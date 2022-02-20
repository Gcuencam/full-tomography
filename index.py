import getopt
import os
import sys

from src.lib.measurement_collector import collect_measurements

os.environ['DEBUG'] = 'False'


def main(argv):
    shots = 0
    qc_size = 0
    output_file_name = 'measurements.txt'
    try:
        opts, args = getopt.getopt(argv, "dhs:q:o:", ["shots=", "qubits=", "output="])
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
    collect_measurements(qc_size, shots, output_file_name)


if __name__ == '__main__':
    main(sys.argv[1:])
