from jax.config import config
import numpy as np

cPRECISION = np.complex64
config.update("jax_enable_x64", cPRECISION.__name__ == 'complex128')
import scipy.optimize as spo

import jax
import jax.numpy as jnp

from .lie import Hamiltonian
from .utils import golden_section_search, commuting_ansatz, prepare_random_initial_parameters, prepare_random_parameters


# Roeland: One thing I recently learned is that you want to make sure that jitted functions are pure functions that
# do not rely on stateful objects, i.e. self.object. Jax will think that these objects are dynamical and slow down.
# You want to return a helper function that feeds in all the static objects and return a function that only depends
# on the non-static inputs.
def get_compute_matrix_fn(commuting_ansatz_matrix, basis):
    def compute_matrix(params):
        commuting_params = jnp.matmul(commuting_ansatz_matrix, params)
        A = jnp.tensordot(commuting_params, basis, axes=[[-1], [0]])
        return jax.scipy.linalg.expm(1j * A)

    return compute_matrix


class Optimizer:
    """
    Handling the optimization of the phi parameters.

    Parameters
    ----------
    target_unitary : np.ndarray
        Target unitary for the optimizer to search towards. 
    basis : basis.Basis
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
    basis : basis.Basis
        The basis of the parameters for the Hamiltonian.
    n_qubits : int
        The number of qubits being optimized over.
    init_parameters : np.ndarray
    max_steps : int
    precision : float
    """

    def __init__(self, target_unitary, full_basis, projected_basis, init_parameters=None, max_steps=1000,
                 precision=0.999, max_step_size=2):
        self.target_unitary = target_unitary
        self.full_basis = full_basis
        self.projected_basis = projected_basis

        self.projected_basis_indices = self.full_basis.project(self.projected_basis)

        self.n_qubits = full_basis.n
        self.init_parameters = init_parameters
        self.max_steps = max_steps
        self.precision = precision
        self.max_step_size = max_step_size

        self.free_indices, self.commuting_ansatz_matrix = commuting_ansatz(target_unitary, full_basis, projected_basis)

        if init_parameters is None:
            self.init_parameters = 2 * np.random.rand(len(full_basis))

        self.parameters = [self.init_parameters]
        self.fidelities = [Hamiltonian(projected_basis,
                                       self._get_ansatz_parameters(self.init_parameters)).fidelity(target_unitary)]
        self.step_sizes = [0]
        self.steps = [0]
        compute_matrix_fn = get_compute_matrix_fn(commuting_ansatz_matrix=self.commuting_ansatz_matrix,
                                                  basis=self.projected_basis.basis)
        self.jac = jax.jacobian(compute_matrix_fn, argnums=0, holomorphic=True)
        self.compute_matrix = jax.jit(compute_matrix_fn)
        self.is_succesful = self.optimize()

    def optimize(self):
        step = 0
        while (self.fidelities[-1] < self.precision) and (step < self.max_steps):
            step += 1
            new_phi_ham, fidelity, step_size = self.update_step(step_count=(step, self.max_steps))
            self.parameters.append(new_phi_ham.parameters)
            self.fidelities.append(fidelity)
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
        free_params = self._get_free_parameters(self.parameters[-1])

        # Step 1: find the geodesic between phi_U and target_V
        gamma = phi_ham.geodesic_hamiltonian(self.target_unitary)

        free_params_c = free_params.astype(cPRECISION)
        dU = self.jac(free_params_c)
        U_dagger = self.compute_matrix(-free_params_c)
        omegas = 1.j * np.transpose(np.tensordot(U_dagger, dU, axes=[[1], [0]]), [2, 0, 1])

        # After contracting, move the parameter derivative axis to the first position
        omega_phis = np.array(
            [Hamiltonian.parameters_from_hamiltonian(omega, self.projected_basis) for i, omega in
             enumerate(omegas)])

        # Step 3: Find a linear combination of Omegas that gives the geodesic and update parameters
        print(omega_phis.shape)
        print(gamma.parameters.shape)
        coeffs = Optimizer.linear_comb_projected_coeffs(omega_phis, gamma.parameters, self.free_indices,
                                                        self.commuting_ansatz_matrix)

        if coeffs is None:
            print(
                f"[{step_count[0]}/{step_count[1]}] Didn't find coefficients for Omega direction; restarting...                                                    ",
                end="\r")
            random_parameters = prepare_random_parameters(self.free_indices, self.commuting_ansatz_matrix)

            new_phi_ham = Hamiltonian(self.full_basis, random_parameters)
            fidelity_new_phi = new_phi_ham.fidelity(self.target_unitary)
            # step_size = spla.norm(new_phi_ham.parameters - phi)
            return new_phi_ham, fidelity_new_phi, 0

        # Step 4: Apply a small push in the right direction to give a new phi
        fidelity_phi, fidelity_new_phi, new_phi_ham, step_size = self._new_phi_golden_section_search(phi_ham, coeffs,
                                                                                                     step_size=self.max_step_size)

        if fidelity_new_phi > self.precision:
            print(
                f"[{step_count[0]}/{step_count[1]}] [Fidelity = {fidelity_new_phi}] A solution!                                                                     ")
        elif fidelity_new_phi > fidelity_phi:
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
        print(expander_matrix.shape)
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
        f = self._construct_fidelity_function(phi_ham, coeffs)
        epsilon, fidelity_new_phi = golden_section_search(f, -step_size, 0, tol=1e-5)
        new_phi_ham = Hamiltonian(self.basis, phi_ham.parameters + (epsilon * coeffs))
        return fidelity_phi, fidelity_new_phi, new_phi_ham, epsilon

    def _new_phi_full(self, phi_ham, coeffs, step_size):
        delta_phi = step_size * coeffs
        new_phi_minus = Hamiltonian(self.basis, phi_ham.parameters - delta_phi)
        new_phi_plus = Hamiltonian(self.basis, phi_ham.parameters + delta_phi)
        if new_phi_minus.fidelity(self.target_unitary) > new_phi_plus.fidelity(self.target_unitary):
            new_phi_ham = new_phi_minus
            sign = -1
        else:
            new_phi_ham = new_phi_plus
            sign = +1
        fidelity_phi = phi_ham.fidelity(self.target_unitary)
        fidelity_new_phi = new_phi_ham.fidelity(self.target_unitary)
        return fidelity_phi, fidelity_new_phi, new_phi_ham, sign * step_size

    def _construct_fidelity_function(self, phi_ham, coeffs):
        def fidelity_f(epsilon):
            phi_h = Hamiltonian(self.full_basis, phi_ham.parameters + (epsilon * coeffs))
            return phi_h.fidelity(self.target_unitary)

        return fidelity_f

    def _get_projected_parameters(self, parameters):
        return parameters[self.projected_basis_indices]

    def _get_ansatz_parameters(self, parameters):
        new_parameters = parameters[self.projected_basis_indices]
        return self.commuting_ansatz_matrix @ new_parameters

    def _get_free_parameters(self, parameters):
        return np.multiply(self.free_indices, self._get_projected_parameters(parameters))
