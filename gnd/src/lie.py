import numpy as np
import scipy.linalg as spla


class Hamiltonian:
    """
    Object representation of the Hamiltonians defined by a Lie algebra basis and phi components.

    Parameters
    ----------
    basis : gnd.Basis
        Generally the PauliBasis object, which defines the basis of the Lie algebra for the phi 
        parameters.
    parameters : np.array
        Numpy array of same length as basis defining the components of the basis.

    Attributes
    ----------
    basis : gnd.PauliBasis
        Basis object including the basis elements and labels.
    parameters : np.array
        Lie algebra component vector.
    matrix : np.ndarray
        Matrix representation of the Hamiltonian.
    n : int 
        The number of qubits that the Hamiltonian acts on is stored.
    """

    def __init__(self, basis, parameters):
        self.basis = basis
        self.parameters = parameters
        self.matrix = self._matrix()
        self.unitary = spla.expm(1.j * self.matrix)

    def geodesic_hamiltonian(self, target_unitary):
        """
        Returns the geodesic to a target unitary.

        Parameters
        ----------
        target_unitary : np.ndarray
            The unitary target

        Returns
        -------
        Hamiltonian
            The geodesic hamiltonian.
        """
        g = -1.j * spla.logm(self.unitary.conj().T @ target_unitary)
        g_params = Hamiltonian.parameters_from_hamiltonian(g, self.basis)
        return Hamiltonian(self.basis, g_params)

    def fidelity(self, unitary):
        """
        Returns the fidelity to a target unitary.

        Parameters
        ----------
        unitary : np.ndarray
            The unitary target

        Returns
        -------
        float
            The fidelity.
        """
        return Unitary.unitary_fidelity(self.unitary, unitary)

    @staticmethod
    def parameters_from_hamiltonian(hamiltonian, basis):
        """
        Returns the parameters from a Hamiltonian.

        Parameters
        ----------
        hamiltonian : np.ndarray
            The Hamiltonian for which we want to find the parameters.
        basis: gnd.Basis
            The basis to find the parameters in.

        Returns
        -------
        np.array
            Parameters vector.
        """
        return np.real(np.einsum("ijk, kj->i", basis.basis, hamiltonian)) / (len(hamiltonian[0]))

    def _matrix(self):
        return np.einsum("ijk,i->jk", self.basis.basis, self.parameters)


class Unitary:
    """
    Object representation of the Unitary defined by a matrix.

    Parameters
    ----------
    unitary : np.ndarray
        Matrix of size 2^n x 2^n that describes a unitary acting on n qubits

    Attributes
    ----------
    matrix : np.ndarray
        Matrix representation of the unitary.
    n : int 
        The number of qubits that the unitary acts on.
    """

    def __init__(self, unitary):
        self.matrix, self.n = self._check_is_unitary(unitary)

    def parameters(self):
        """
        Calculate parameters from the unitary using the principal logarithm.

        Parameters
        ----------

        Returns
        -------

        """
        pass

    @staticmethod
    def unitary_fidelity(unitary1, unitary2):
        return np.abs(np.trace(unitary1.conj().T @ unitary2)) / len(unitary1[0])

    def _check_is_unitary(unitary):
        if not np.allclose(np.eye(len(unitary)), unitary @ unitary.T.conj()):
            raise ValueError("Matrix given to Unitary must be unitary: U U^dagger = U^dagger U = I")
        if (not np.log2(len(unitary)).is_integer()) or (not unitary.shape[0] == unitary.shape[1]):
            raise ValueError("Matrix given to Unitary must be size 2^n x 2^n")
        return unitary, int(np.log2(len(unitary)))

    @staticmethod
    def unitary_fidelity(unitary1, unitary2):
        return np.abs(np.trace(unitary1.conj().T @ unitary2)) / len(unitary1[0])

