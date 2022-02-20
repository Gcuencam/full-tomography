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

## How to run the code
```shell
# -s is the number of shots that each circuit will be executed
# -q is the size of the circuits in qbits
# -o is the output filename
# -d is to enable debug mode
python3 index.py -s 10 -q 4 -o test.txt -d
```

#### Other possible issues
#### Installation for mac M1
```shell
pip install "git+https://github.com/Qiskit/qiskit-aer.git"
```