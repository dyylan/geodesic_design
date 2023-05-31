import numpy as np

from gnd.utils import multikron

I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])

class ToffoliConfig:
    def __init__(self):
        self.nqubits = 3
        self.unitary = np.array([[1,0,0,0,0,0,0,0],
                                 [0,1,0,0,0,0,0,0],
                                 [0,0,1,0,0,0,0,0],
                                 [0,0,0,1,0,0,0,0],
                                 [0,0,0,0,1,0,0,0],
                                 [0,0,0,0,0,1,0,0],
                                 [0,0,0,0,0,0,0,1],
                                 [0,0,0,0,0,0,1,0]])
        self.precision = 0.999
        self.max_steps = 1000
        self.seed = 1

    def __str__(self):
        return "toffoli"

    def __dir__(self):
        return ["precision", "max_steps", "seed"]


class FredkinConfig:
    def __init__(self):
        self.nqubits = 3
        self.unitary = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1]])
        self.precision = 0.999
        self.max_steps = 1000
        self.seed = 1

    def __str__(self):
        return "fredkin"

    def __dir__(self):
        return ["precision", "max_steps", "seed"]


class QFTqubitConfig:
    def __init__(self):
        self.nqubits = 3
        w = np.exp(1.j * 2 * np.pi / 2**self.nqubits)
        self.unitary = (1/np.sqrt(2**self.nqubits)) * np.array([[w**(i*j) for i in range(2**self.nqubits)] for j in range(2**self.nqubits)])
        self.precision = 0.999
        self.max_steps = 1000
        self.seed = 1

    def __str__(self):
        return f"QFT_{self.nqubits}_qubits"

    def __dir__(self):
        return ["precision", "max_steps", "seed"]
    

class Weight2ParityZConfig:
    def __init__(self):
        self.nqubits = 3
        self.unitary = (multikron([I,I,I]) + multikron([Z,Z,I]) + multikron([I,I,X]) - multikron([Z,Z,X]))/2
        self.precision = 0.999
        self.max_steps = 1000
        self.seed = 1

    def __str__(self):
        return f"weight_2_parity_check_Z"

    def __dir__(self):
        return ["precision", "max_steps", "seed"]


class Weight4ParityZConfig:
    def __init__(self):
        self.nqubits = 5
        self.unitary = (multikron([I,I,I,I,I]) + multikron([Z,Z,Z,Z,I]) + multikron([I,I,I,I,X]) - multikron([Z,Z,Z,Z,X]))/2
        self.precision = 0.999
        self.max_steps = 1000
        self.seed = 1

    def __str__(self):
        return f"weight_4_parity_check_Z"

    def __dir__(self):
        return ["precision", "max_steps", "seed"]
