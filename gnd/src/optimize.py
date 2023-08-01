from jax.config import config
import numpy as np

config.update("jax_enable_x64", True)
import scipy.optimize as spo

import jax
import jax.numpy as jnp

from .lie import Hamiltonian
from .utils import golden_section_search, commuting_ansatz, prepare_random_parameters


# Roeland: One thing I recently learned is that you want to make sure that jitted functions are pure functions that
# do not rely on stateful objects, i.e. self.object. Jax will think that these objects are dynamical and slow down.
# You want to return a helper function that feeds in all the static objects and return a function that only depends
# on the non-static inputs.
def get_compute_matrix_fn(commuting_ansatz_matrix: np.ndarray, basis: np.ndarray):
    def compute_matrix(params):
        commuting_params = jnp.matmul(commuting_ansatz_matrix, params)
        A = jnp.tensordot(commuting_params, basis, axes=[[-1], [0]])
        return jax.scipy.linalg.expm(1j * A)

    return compute_matrix


def get_project_omegas_fn(basis: np.ndarray, dim: int):
    def project_omegas(x):
        return jnp.real(jnp.einsum("ijk, nkj->ni", basis, x)) / dim

    return project_omegas


def get_Udagger_dU_contraction_fn():
    def Udagger_dU_contraction(x, y):
        return 1j * jnp.transpose(jnp.tensordot(x, y, axes=[[1], [0]]), [2, 0, 1])

    return Udagger_dU_contraction


def get_fidelity_fn(basis: np.ndarray, target: np.ndarray):
    def fidelity(epsilon, x, y):
        phi_h = jnp.einsum("ijk,i->jk", basis, x + epsilon * y)
        unitary = jax.scipy.linalg.expm(1j * phi_h)
        return jnp.abs(jnp.trace(target.conj().T @ unitary)) / len(target[0])

    return fidelity


class Optimizer:
    """
    Handling the optimization of the phi parameters.

    Parameters
    ----------
    target_unitary : np.ndarray
        Target unitary for the optimizer to search towards.
    full_basis : basis.Basis
        Generally basis.PauliBasis object, defines the basis of the Lie algebra for the phi
        parameters
    init_parameters : np.ndarray
        The initial parameters of the Hamiltonian to be optimized.
    max_steps : int, default=1000
        Number of steps before optimization is halted.
    precision : float, default=0.999
        Precision of before a final solution is accepted.

    Attributes
    ----------
    target_unitary : np.ndarray
        Stores the target unitary as a unitary object.
    full_basis : basis.Basis
        The basis of the parameters for the Hamiltonian.
    projected_basis : basis.Basis
        The basis of the restricted Hamiltonian
    init_parameters : np.ndarray
    max_steps : int
    precision : float
    """

    def __init__(self, target_unitary, full_basis, projected_basis, init_parameters=None, max_steps=1000,
                 precision=0.999, max_step_size=2, commute=True):
        self.target_unitary = target_unitary
        self.full_basis = full_basis
        self.projected_basis = projected_basis
        self.init_parameters = init_parameters
        self.max_steps = max_steps
        self.precision = precision
        self.max_step_size = max_step_size
        # Get the projected and free indices in the full space
        self.projected_indices = np.array(projected_basis.overlap(full_basis), dtype=bool)
        if commute:
            self.free_indices, self.commuting_ansatz_matrix = commuting_ansatz(target_unitary, full_basis,
                                                                               self.projected_indices)
        else:
            self.free_indices = self.projected_indices
            self.commuting_ansatz_matrix = np.identity(len(self.free_indices))

        # Get the free indices within the projected space
        indices = np.where(self.projected_indices)[0]
        free_indices = np.where(self.free_indices)[0]
        locs = np.array([np.where(indices == idx)[0] for idx in free_indices])
        self.free_indices_small = np.zeros(len(indices), dtype=int)
        self.free_indices_small[locs] = 1
        # Get the matrix that gives us the projection parameters within the projected space
        self.commuting_ansatz_matrix_free = self.commuting_ansatz_matrix[self.projected_indices, :][:,
                                            self.projected_indices]
        # Initialize variables
        if init_parameters is None:
            self.init_parameters = prepare_random_parameters(self.free_indices, self.commuting_ansatz_matrix)
        self.parameters = [self.init_parameters]
        self.fidelities = [Hamiltonian(full_basis, self.init_parameters).fidelity(target_unitary)]
        self.step_sizes = [0]
        self.steps = [0]
        # Get the Jax functions from the helper functions
        compute_matrix_fn = get_compute_matrix_fn(commuting_ansatz_matrix=self.commuting_ansatz_matrix_free,
                                                  basis=self.projected_basis.basis)
        project_omegas_fn = get_project_omegas_fn(self.full_basis.basis, self.full_basis.dim)
        Udagger_dU_contraction = get_Udagger_dU_contraction_fn()
        fidelity = get_fidelity_fn(self.projected_basis.basis, self.target_unitary)
        # Jit all the Jax functions
        self.jac = jax.jacobian(compute_matrix_fn, argnums=0, holomorphic=True)
        self.compute_matrix = jax.jit(compute_matrix_fn)
        self.project_omegas = jax.jit(project_omegas_fn)
        self.Udagger_dU_contractionn = jax.jit(Udagger_dU_contraction)
        self.fidelity = jax.jit(fidelity)
        # Start optimization
        self.is_succesful = self.optimize()

    def optimize(self):
        step = 0
        while (self.fidelities[-1] < self.precision) and (step < self.max_steps):
            step += 1
            new_phi_ham, fidelity, step_size = self.update_step(step_count=(step, self.max_steps))
            self.parameters.append(new_phi_ham.parameters)
            self.fidelities.append(fidelity)
            print(fidelity)
            self.step_sizes.append(step_size)
            self.steps.append(step)
        print("")
        if self.fidelities[-1] >= self.precision:
            return True
        else:
            return False

    def update_step(self, step_count=(None, None)):
        # Step 0: find the unitary from phi
        phi = self.parameters[-1]
        phi_ham = Hamiltonian(self.full_basis, phi)
        free_params = np.multiply(self.free_indices, self.parameters[-1])

        # Step 1: find the geodesic between phi_U and target_V
        gamma = phi_ham.geodesic_hamiltonian(self.target_unitary)

        free_params_c = free_params[self.projected_indices].astype(np.complex128)

        dU = self.jac(free_params_c)
        U_dagger = self.compute_matrix(-free_params_c)

        omegas = self.Udagger_dU_contractionn(U_dagger, dU)

        # After contracting, move the parameter derivative axis to the first position
        omega_phis = self.project_omegas(omegas)

        # Step 3: Find a linear combination of Omegas that gives the geodesic and update parameters
        temp_coeffs = Optimizer.linear_comb_projected_coeffs(omega_phis, gamma.parameters, self.free_indices_small,
                                                      self.commuting_ansatz_matrix_free)
        # Expand the coefficients

        if temp_coeffs is None:
            print(
                f"[{step_count[0]}/{step_count[1]}] Didn't find coefficients for Omega direction; restarting...                                                    ",
                end="\r")
            random_parameters = prepare_random_parameters(self.free_indices, self.commuting_ansatz_matrix)

            new_phi_ham = Hamiltonian(self.full_basis, random_parameters)
            fidelity_new_phi = new_phi_ham.fidelity(self.target_unitary)

            return new_phi_ham, fidelity_new_phi, 0

        # Expand the coefficients to the larger space
        coeffs = np.zeros_like(phi_ham.parameters)
        coeffs[self.projected_indices] = temp_coeffs

        # Step 4: Apply a small push in the right direction to give a new phi
        fidelity_phi, fidelity_new_phi, new_phi_ham, step_size = self._new_phi_golden_section_search(phi_ham, coeffs,
                                                                                                     step_size=self.max_step_size)

        if fidelity_new_phi > self.precision:
            print(
                f"[{step_count[0]}/{step_count[1]}] [Fidelity = {fidelity_new_phi}] A solution!                                                                     ")
        elif (fidelity_new_phi > fidelity_phi) and not np.isclose(fidelity_new_phi, fidelity_phi, atol=(1 - self.precision) / 100):
            print(
                f"[{step_count[0]}/{step_count[1]}] [Fidelity = {fidelity_new_phi}] Omega geodesic gave a positive fidelity update for this step...                 ",
                end="\r")
        else:
            print(
                f"[{step_count[0]}/{step_count[1]}] [Fidelity = {fidelity_phi}] Omega geodesic gave a negative fidelity update for this step. Moving phi away...    ",
                end="\r")
            proj_c = prepare_random_parameters(self.free_indices, self.commuting_ansatz_matrix)

            # Use the Gram-Schmidt procedure to generate a perpendicular vector to the previous coefficients.
            proj_c = proj_c - (((proj_c @ coeffs) / (coeffs @ coeffs)) * coeffs)

            fidelity_phi, fidelity_new_phi, new_phi_ham, step_size = self._new_phi_full(phi_ham, proj_c, step_size=1)

        return new_phi_ham, fidelity_new_phi, step_size

    @staticmethod
    def linear_comb_projected_coeffs(combination_vectors, target_vector, projected_indices, commuting_ansatz):
        num_params = sum(projected_indices)
        expander_matrix = np.identity(num_params)
        for i, index in enumerate(projected_indices):
            if not index:
                expander_matrix = np.insert(expander_matrix, i, np.zeros(num_params), axis=0)
        res = spo.least_squares(
            lambda x: combination_vectors.T @ commuting_ansatz @ expander_matrix @ x - target_vector,
            x0=np.zeros(num_params),
            method='lm')
        if not res.success:
            return None
        else:
            coeffs = commuting_ansatz @ expander_matrix @ res.x
        return coeffs

    def _new_phi_golden_section_search(self, phi_ham, coeffs, step_size):
        fidelity_phi = phi_ham.fidelity(self.target_unitary)
        f = lambda x: self.fidelity(x, phi_ham.parameters[self.projected_indices], coeffs[self.projected_indices])
        epsilon, fidelity_new_phi = golden_section_search(f, -step_size, 0, tol=1e-5)
        new_phi_ham = Hamiltonian(self.full_basis, phi_ham.parameters + (epsilon * coeffs))
        return fidelity_phi, fidelity_new_phi, new_phi_ham, epsilon

    def _new_phi_full(self, phi_ham, coeffs, step_size): #TODO could jitify this but probably not necessary
        delta_phi = step_size * coeffs
        new_phi_minus = Hamiltonian(self.full_basis, phi_ham.parameters - delta_phi)
        new_phi_plus = Hamiltonian(self.full_basis, phi_ham.parameters + delta_phi)
        if new_phi_minus.fidelity(self.target_unitary) > new_phi_plus.fidelity(self.target_unitary):
            new_phi_ham = new_phi_minus
            sign = -1
        else:
            new_phi_ham = new_phi_plus
            sign = +1
        fidelity_phi = phi_ham.fidelity(self.target_unitary)
        fidelity_new_phi = new_phi_ham.fidelity(self.target_unitary)
        return fidelity_phi, fidelity_new_phi, new_phi_ham, sign * step_size
