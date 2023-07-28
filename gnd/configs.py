import numpy as np
from src.utils import multikron

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])


class CnotConfig:
    def __init__(self):
        self.nqubits = 2
        self.unitary = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]])
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return "toffoli"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class ToffoliConfig:
    def __init__(self):
        self.nqubits = 3
        self.unitary = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 1, 0]])
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return "toffoli"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class CxNotConfig:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        N = 2**nqubits
        self.unitary = np.array(
            [[1 if (i == j and i < N-2) or (i == N-2 and j == N-1) or (i == N-1 and j == N-2) else 0 for i in range(N)] for j in range(N)]
        )
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return f"C{self.nqubits}Not"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


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
        self.commute = True
        self.seed = 1

    def __str__(self):
        return "fredkin"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class QFTqubitConfig:
    def __init__(self):
        self.nqubits = 3
        w = np.exp(1.j * 2 * np.pi / 2 ** self.nqubits)
        self.unitary = (1 / np.sqrt(2 ** self.nqubits)) * np.array(
            [[w ** (i * j) for i in range(2 ** self.nqubits)] for j in range(2 ** self.nqubits)])
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return f"QFT_{self.nqubits}_qubits"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class Weight2ParityZConfig:
    def __init__(self):
        self.nqubits = 3
        self.unitary = (multikron([I, I, I]) + multikron([Z, Z, I]) + multikron([I, I, X]) - multikron([Z, Z, X])) / 2
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return f"w2pz"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class Weight2ParityXConfig:
    def __init__(self):
        self.nqubits = 3
        self.unitary = (multikron([I, I, I]) + multikron([X, X, I]) + multikron([I, I, X]) - multikron([X, X, X])) / 2
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return f"w2px"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class Weight3ParityZConfig:
    def __init__(self):
        self.nqubits = 4
        self.unitary = (multikron([I, I, I, I]) + multikron([Z, Z, Z, I]) + multikron([I, I, I, X]) - multikron([Z, Z, Z, X])) / 2
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return f"w3pz"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class Weight3ParityXConfig:
    def __init__(self):
        self.nqubits = 4
        self.unitary = (multikron([I, I, I, I]) + multikron([X, X, X, I]) + multikron([I, I, I, X]) - multikron([X, X, X, X])) / 2
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return f"w3px"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class Weight4ParityZConfig:
    def __init__(self):
        self.nqubits = 5
        self.unitary = (multikron([I, I, I, I, I]) + multikron([Z, Z, Z, Z, I]) + multikron(
            [I, I, I, I, X]) - multikron([Z, Z, Z, Z, X])) / 2
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return f"w4pz"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]


class Weight4ParityXConfig:
    def __init__(self):
        self.nqubits = 5
        self.unitary = (multikron([I, I, I, I, I]) + multikron([X, X, X, X, I]) + multikron(
            [I, I, I, I, X]) - multikron([X, X, X, X, X])) / 2
        self.precision = 0.999
        self.max_steps = 1000
        self.commute = True
        self.seed = 1

    def __str__(self):
        return f"w4px"

    def __dir__(self):
        return ["precision", "max_steps", "commute", "seed"]
