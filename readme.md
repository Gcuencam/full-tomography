# Quantum tomography

## How to install and create virtual environment
[Virtual environments reference](https://virtualenv.pypa.io/en/latest/)
```shell
virtualenv venv
```

## How to install dependencies
```shell
pip install -r requirements.txt
```

## How to activate virtual environment
```sql
source venv/bin/activate
```

## How to run qiskit state tomography
```shell
python3 qiskit_qst.py --q_bits_lower_limit=2 --q_bits_upper_limit=5 --shots_lower_limit=100 --shots_upper_limit=1000 --shots_pace=50 --output_folder=new_data5qbits1000shots
```

## How to run the code
```shell
# -s is the number of shots that each circuit will be executed
# -q is the size of the circuits in qbits
# -o is the output filename
# -d is to enable debug mode
# -m measuremente type. pauli or povm
# -t prepared state type. w or ghz
python3 index.py -s 10 -q 4 -o test.txt -d -m [pauli | povm] -t [w | ghz]
```

#### Other possible issues
#### Installation for mac M1
```shell
pip install "git+https://github.com/Qiskit/qiskit-aer.git"
```