import numpy as np
import cvxpy as cp
import scipy.linalg as spla
import pennylane as qml

from .lie import Hamiltonian
from .utils import golden_section_search


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
    def __init__(self, target_unitary, basis, init_parameters, max_steps=1000, precision=0.999):
        self.target_unitary = target_unitary
        self.basis = basis
        self.n_qubits = basis.n
        self.init_parameters = init_parameters
        self.max_steps = max_steps
        self.precision = precision

        self.parameters = [init_parameters]
        self.fidelities = [Hamiltonian(basis, init_parameters).fidelity(target_unitary)]
        self.step_sizes = [0]
        self.steps = [0]
        self.is_succesful = self.optimize()

    def optimize(self):
        step = 0
        while (self.fidelities[-1] < self.precision) and (step < self.max_steps):
            step += 1
            new_phi_ham, fidelity, step_size = self.update_step(step_count=(step,self.max_steps))
            self.parameters.append(new_phi_ham.parameters)
            self.fidelities.append(fidelity)
            self.step_sizes.append(step_size)
            self.steps.append(step)
        if self.fidelities[-1] >= self.precision:
            return True
        else:
            return False

    def update_step(self, step_count=(None,None)):
        # Step 0: find the unitary from phi
        phi = self.parameters[-1]
        phi_ham = Hamiltonian(self.basis, phi)
        
        # Step 1: find the geodesic between phi_U and target_V
        gamma = phi_ham.geodesic_hamiltonian(self.target_unitary)
        
        # Step 2: find the Omegas
        projected_indices = self.basis.two_body_projection_indices()

        su = qml.SpecialUnitary(phi, [i for i in range(self.n_qubits)])
        omegas = 1.j * su.get_one_parameter_generators(interface="jax")

        omega_phis = np.array([projected_indices[i] * Hamiltonian.parameters_from_hamiltonian(omega, self.basis) for i, omega in enumerate(omegas)])

        # Step 3: Find a linear combination of Omegas that gives the geodesic and update parameters
        coeffs = Optimizer.linear_comb_projected_coeffs(omega_phis, gamma.parameters, projected_indices)
        
        if coeffs is None:
            print(f"[{step_count[0]}/{step_count[1]}] Didn't find coefficients for Omega direction; restarting...                                                     ", end="\r")
            new_phi_ham = Hamiltonian(self.basis, self.basis.two_body_projection(2*np.random.rand(len(self.basis.basis)) - 1))
            fidelity_new_phi = new_phi_ham.fidelity(self.target_unitary)
            # step_size = spla.norm(new_phi_ham.parameters - phi)
            return new_phi_ham, fidelity_new_phi, 0

        # Step 4: Apply a small push in the right direction to give a new phi
        fidelity_phi, fidelity_new_phi, new_phi_ham, step_size = self._new_phi_golden_section_search(phi_ham, coeffs, step_size=2)

        if fidelity_new_phi > self.precision:
#             print(f"\n[{step_count[0]}/{step_count[1]}] Solution found!")
            print(f"[{step_count[0]}/{step_count[1]}] [Fidelity = {fidelity_new_phi}] A solution!                                                                     ")
#         elif fidelity_new_phi - ((1-precision)/100) > fidelity_phi:
        elif fidelity_new_phi > fidelity_phi:
            print(f"[{step_count[0]}/{step_count[1]}] [Fidelity = {fidelity_new_phi}] Omega geodesic gave a positive fidelity update for this step...                 ", end="\r")
        else:
            print(f"[{step_count[0]}/{step_count[1]}] [Fidelity = {fidelity_phi}] Omega geodesic gave a negative fidelity update for this step. Moving phi away...    ", end="\r")
            c = 2*np.random.rand(len(self.basis.basis)) - 1
            proj_c = np.multiply(projected_indices, c)

            # Use the Gram-Schmidt procedure to generate a perpendicular vector to the previous coefficients.
            proj_c = proj_c - (((proj_c @ coeffs)/(coeffs @ coeffs)) * coeffs)

            fidelity_phi, fidelity_new_phi, new_phi_ham, step_size = self._new_phi_full(phi_ham, proj_c, step_size=1)
        
        return new_phi_ham, fidelity_new_phi, step_size

    @staticmethod
    def linear_comb_projected_coeffs(combination_vectors, target_vector, projected_indices):
        delete_indices = np.where(projected_indices == 0)[0]
        combination_vectors_projected = np.delete(combination_vectors, delete_indices, axis=0)
        x = cp.Variable(combination_vectors_projected.shape[0])
        cost = cp.sum_squares(combination_vectors.T @ x - target_vector)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()
        if x.value is None:
            return x.value
        else:
            coeffs = np.insert(x.value, delete_indices, values=0, axis=0)
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
        return fidelity_phi, fidelity_new_phi, new_phi_ham, sign*step_size

    def _construct_fidelity_function(self, phi_ham, coeffs):
        def fidelity_f(epsilon):
            phi_h = Hamiltonian(self.basis, phi_ham.parameters+(epsilon * coeffs))
            return phi_h.fidelity(self.target_unitary)
        return fidelity_f