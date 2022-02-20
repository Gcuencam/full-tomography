def buildWState(circuit, referencePosition: int, n: int):
    if (n < 2):
        raise Exception('The size must be at least two.')
    if (referencePosition < 0):
        raise Exception('The reference position must be at least zero.')
    if (n > circuit.width() / 2):
        raise Exception('The circuit does not have enough qubits.')
    if (referencePosition + n > circuit.width() / 2):
        raise Exception('The reference position must be lower.')
    firstPosition = referencePosition
    lastPosition = referencePosition + n - 1
    _circuit = circuit.copy()
    for i in range(firstPosition, lastPosition + 1):
        _circuit.reset(i)
    if (n >= 3):
        _circuit.ry(2 * np.arccos(1 / m.sqrt(n)), firstPosition)
        for i in range(n - 3):
            _circuit.cry(2 * np.arccos(1 / m.sqrt(n - (i + 1))), firstPosition + i, firstPosition + i + 1)
    if (n == 2):
        _circuit.h(firstPosition)
    else:
        _circuit.ch(lastPosition - 2, lastPosition - 1)
    for i in reversed(range(firstPosition, lastPosition)):
        _circuit.cx(i, i + 1)
    _circuit.x(firstPosition)
    return _circuit
