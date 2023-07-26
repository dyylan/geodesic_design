import numpy as np
import itertools as it


class Basis:
    def __init__(self, basis: np.ndarray):
        assert basis.ndim == 3, '`basis` must be a rank 3 tensor'
        assert (basis.shape[1] == basis.shape[2]) and (np.log2(basis.shape[1]) == int(np.log2(basis.shape[1]))), \
            '`basis` must be a tensor of shape (n, 2**n, 2**n), where n corresponds to the matrix dimension, ' \
            f'received {basis.shape}'
        # Get all dot products A_i @ A_j
        outer = np.einsum('nij,mjk->nmik', basis, basis)
        # Get the trace of all dot products
        traces = np.einsum('ijkk->ij', outer)
        assert np.allclose(np.diag(np.diag(traces)), traces), 'basis is not orthogonal with respect to the trace'
        self._basis = basis
        self._n = int(np.log2(basis.shape[1]))
        assert self._n

    def linear_span(self, parameters):
        parameters = np.reshape(parameters, (-1, 1, 1))
        return np.einsum('nij,nij->ij', parameters, self._basis)

    def project(self, other):
        # Get all dot products A_i @ A_j
        outer = np.einsum('nij,mjk->nmik', self._basis, other._basis)
        # Get the trace of all dot products
        traces = np.einsum('ijkk->ij', outer)
        return ~np.isclose(np.sum(traces, axis=1), 0)
    @property
    def basis(self):
        return self._basis

    @property
    def n(self):
        return self._n

    @property
    def shape(self):
        return self._basis.shape

    def __len__(self):
        return self._basis.shape[0]


def construct_two_body_pauli_basis(n: int):
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    b = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.
        if len(np.nonzero(comb)[0]) <= 2:
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

    return Basis(np.stack(b))


def construct_full_pauli_basis(n: int):
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    labels = []
    b = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.
        labels.append(comb)
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

    return Basis(np.stack(b))

# if __name__ == '__main__':
#     b = construct_full_pauli_basis(2)
#     b = construct_two_body_pauli_basis(3)
#     print(b.basis.shape)
#     p = np.random.randn(b.basis.shape[0])
#     print(b.linear_span(p))