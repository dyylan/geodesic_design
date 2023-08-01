import numpy as np
import jax.numpy as jnp
import jax
import itertools as it


@jax.jit
def trace_dot_jit(x, y):
    return jnp.trace(x @ y)


def traces(b_1, b_2):
    indices = []
    len_1 = b_1.shape[0]
    len_2 = b_2.shape[0]
    for i in range(len_1):
        for j in range(len_2):
            indices.append([i, j])
    indices = np.stack(indices)
    jself = jnp.array(b_1)
    jother = jnp.array(b_2)
    carry = jnp.empty((len_1, len_2), dtype=complex)

    def scan_body(c, idx):
        idx, jdx = idx
        c = c.at[idx, jdx].set(trace_dot_jit(jself[idx], jother[jdx]))
        return c, None

    carry, _ = jax.lax.scan(scan_body, init=carry, xs=indices)
    return carry


class Basis:
    def __init__(self, basis: np.ndarray):
        assert basis.ndim == 3, '`basis` must be a rank 3 tensor'
        assert (basis.shape[1] == basis.shape[2]) and (np.log2(basis.shape[1]) == int(np.log2(basis.shape[1]))), \
            '`basis` must be a tensor of shape (n, 2**n, 2**n), where n corresponds to the matrix dimension, ' \
            f'received {basis.shape}'
        self._basis = basis
        self._dim = basis.shape[1]
        self._n = int(np.log2(basis.shape[1]))
        assert self._n

    def linear_span(self, parameters):
        parameters = np.reshape(parameters, (-1, 1, 1))
        return np.einsum('nij,nij->ij', parameters, self._basis)

    def overlap(self, other):
        out = traces(self.basis, other.basis)
        return ~np.isclose(np.sum(out, axis=0), 0)

    def verify(self):
        out = traces(self.basis, self.basis)
        return np.allclose(np.diag(np.diag(out)), out)

    @property
    def basis(self):
        return self._basis

    @property
    def n(self):
        return self._n

    @property
    def dim(self):
        return self._dim

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
#     n = 6
#     b = construct_full_pauli_basis(n)
#     b_proj = construct_two_body_pauli_basis(n)
#     print(b.shape)
#     print(b_proj.shape)
#     print(b_proj.overlap(b))
#     print(b.overlap(b_proj))
