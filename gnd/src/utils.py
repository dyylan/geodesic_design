import numpy as np
import scipy.linalg as spla
import sympy

from src import lie

invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2


def prepare_random_initial_parameters(target_unitary, basis):
    randoms = 2*np.random.rand(len(basis.basis))-1
    proj_indices, mat = commuting_ansatz(target_unitary, basis, basis.two_body_projection_indices())
    init_parameters = mat @ np.multiply(proj_indices, randoms)
    return init_parameters

def prepare_random_parameters(proj_indices, commuting_matrix):
    randoms = 2*np.random.rand(len(proj_indices))
    parameters = commuting_matrix @ np.multiply(proj_indices, randoms)
    return parameters

def commuting_ansatz(target_unitary, basis, projected_indices):
    ham = -1.j * spla.logm(target_unitary)
    target_params = lie.Hamiltonian.parameters_from_hamiltonian(ham, basis)
    target_ham = lie.Hamiltonian(basis, target_params)
    h_params = [sympy.Symbol(f'h_{i}') if ind else 0 for i, ind in enumerate(projected_indices)]
    h_mat = None
    for i, b in enumerate(target_ham.basis.basis):
        if h_mat is None:
            h_mat = h_params[i] * sympy.Matrix(b)
        else:
            h_mat += h_params[i] * sympy.Matrix(b)
    # h_mat = sum([h_params[i] * sympy.Matrix(b) for i, b in enumerate(target_ham.basis.basis)])
    # sols = sympy.solve([h_mat * target_ham.matrix - target_ham.matrix * h_mat, sum(h_params_squared)-1])
    sols = sympy.solve(h_mat * target_ham.matrix - target_ham.matrix * h_mat)
    indices = remove_solution_free_parameters(h_params, sols)
    mat = construct_commuting_ansatz_matrix(h_params, sols)
    return indices, mat

def construct_commuting_ansatz_matrix(params, sols):
    mat = np.zeros((len(params), len(params)))
    for j, h in enumerate(params):
        if h:
            h_sub = {m : 0 for m in params if m}
            h_sub[h] = 1
            for i, s in enumerate(params):
                if i == j:
                    mat[i, j] = 1
                if s in sols:
                    mat[i, j] = sols[s].subs(h_sub)
    return mat

def remove_solution_free_parameters(params, sols):
    indices = [0 if h in sols else 1 if h else 0 for h in params]
    return indices

def multikron(matrices):
    product = matrices[0]
    for mat in matrices[1:]:
        product = np.kron(product, mat)
    return product

def unitary_fidelity(unitary1, unitary2):
    return np.abs(np.trace(unitary1.conj().T @ unitary2))/len(unitary1[0])

def golden_section_search(f, a, b, tol=1e-5):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.
    
    Example:
    f = lambda x: (x-2)**2
    a = 1
    b = 5
    tol = 1e-5
    (c,d) = gss(f, a, b, tol)
    print(c, d)
    1.9999959837979107 2.0000050911830893
    source: https://en.wikipedia.org/wiki/Golden-section_search
    """

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)
    
    for k in range(n-1):
        if yc > yd: 
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
    if yc < yd:
        return c, yc
    else:
        return d, yd