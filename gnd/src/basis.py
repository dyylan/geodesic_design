import numpy as np
import itertools as it


class PauliBasis:
    def __init__(self, n):
        self.n = n
        self._construct_basis()

    def two_body_projection_indices(self):
        indices = []
        for b in self.labels:
            if len(np.nonzero(b)[0]) <= 2:
                indices.append(1)
            else:
                indices.append(0)
        return indices

    def two_body_projection(self, parameters):
        indices = self.two_body_projection_indices()
        return np.array([t * indices[i] for i, t in enumerate(parameters)])

    def _construct_basis(self):
        I = np.eye(2).astype(complex)
        X = np.array([[0, 1], [1, 0]], complex)
        Y = np.array([[0, -1j], [1j, 0]], complex)
        Z = np.array([[1, 0], [0, -1]], complex)
        self.labels = []
        b = []
        for comb in list(it.product([0, 1, 2, 3], repeat=self.n))[1:]:
            p = 1.
            self.labels.append(comb)
            for c in comb:
                if c == 0:
                    p = np.kron(p, I)
                elif c == 1:
                    p = np.kron(p, X)
                elif c == 2:
                    p = np.kron(p, Y)
                elif c == 3:
                    p = np.kron(p, Z)
            b.append(p)
        self.basis = np.stack(b)