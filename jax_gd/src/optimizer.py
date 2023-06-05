import jax
import jax.numpy as jnp
import numpy as np


class GD(object):
    def __init__(self, learning_rate, grad_fn):
        self.learning_rate = learning_rate
        self.grad_fn = grad_fn

    def update_params(self, params):
        loss, gradient = self.grad_fn(params)
        return loss, params - self.learning_rate * gradient


class ADAM(object):
    def __init__(self, learning_rate: float, grad_fn, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
        self.learning_rate = learning_rate
        self.grad_fn = grad_fn
        self._momentums = 0.0
        self._velocities = 0.0
        self.iterations = 0

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def update_params(self, params):
        beta_1_power = self.beta_1 ** (self.iterations + 1)
        beta_2_power = self.beta_2 ** (self.iterations + 1)
        alpha = self.learning_rate * jnp.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        loss, gradient = self.grad_fn(params)
        self._momentums += (gradient - self._momentums) * (1 - self.beta_1)
        self._velocities += (gradient ** 2 - self._velocities) * (1 - self.beta_2)
        update_step = (self._momentums * alpha) / (jnp.sqrt(self._velocities) + self.epsilon)
        return loss, params - update_step


def unitary_fidelity(u: jnp.array, v: jnp.array) -> float:
    """
    Calculate the fidelity F(U,V) between two unitaries.

    Args:
        u: ndarray of size (n,n) corresponding to the first unitary
        v: ndarray of size (n,n) corresponding to the second unitary

    Returns:
        Real scalar number corresponding to 0 <= F(U,V) <=1
    """
    return jnp.abs(jnp.trace(u.conj().T @ v)) / u.shape[0]


def get_unitary(params: jnp.array, basis: jnp.array) -> jnp.array:
    """
    Construct a unitary via the exponential map

    Args:
        params: ndarray of size (d) corresponding the parameters in the (projected) basis
        basis: ndarray of size (d,n,n) corresponding to the d elements in the basis of (n,n) matrices

    Returns:
        Unitary operator corresponding to exp(i sum_i params_i basis_i)
    """
    H = jnp.einsum("ijk,i->jk", basis, params)
    return jax.scipy.linalg.expm(1.j * H)


def loss_fn(params: jnp.array, target: jnp.array, basis: jnp.array) -> float:
    """

    Args:
        params: ndarray of size (d) corresponding the parameters in the (projected) basis
        target: ndarray of size (n,n) corresponding the target unitary
        basis: ndarray of size (d,n,n) corresponding to the d elements in the basis of (n,n) matrices

    Returns:
        Real scalar corresponding to the fidelity F(target, U(params)

    """
    unitary = get_unitary(params, basis)
    return 1 - unitary_fidelity(target, unitary)


def optimizer_unitary(target: jnp.array, basis: jnp.array, parameters: jnp.array,
                      max_steps: int = 1000, learning_rate: float = 1e-1, precision: float = 0.999,
                      optimizer="gd") -> dict:
    """

    Args:
        target: ndarray of size (n,n) corresponding the target unitary
        basis: ndarray of size (d,n,n) corresponding to the d elements in the basis of (n,n) matrices
        parameters: ndarray of size (d) corresponding the parameters in the (projected) basis
        max_steps: integer corresponding to the maximum number of optimization steps
        learning_rate: float corresponding to the learning rate of the optimizer
        precision: float corresponding to the final fidelity precision we are interested in
        optimizer: string indicating which optimizer to use. Possible values are "gd" and "adam"

    Returns:
        dictionary with fields 'fidelities' and 'parameters' containing the fidelities and parameters at each
        optimization step.

    """
    assert 0.0 <= precision <= 1., f"`precision must be between 0 and 1, received {precision}"
    grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    loss, _ = grad_fn(parameters, target, basis)
    if optimizer == 'gd':
        opt = GD(learning_rate, lambda x: grad_fn(x, target, basis))
    elif optimizer == 'adam':
        opt = ADAM(learning_rate, lambda x: grad_fn(x, target, basis))
    else:
        raise NotImplementedError

    parameters_total = [np.copy(parameters)]
    fidelities_total = [1 - float(loss)]
    print(f"Initial fidelity {loss}")
    for step in range(max_steps):
        loss, parameters = opt.update_params(parameters)
        parameters_total.append(np.copy(parameters))
        fidelities_total.append(1 - float(loss))
        if not (step + 1) % 100:
            print(f"Step {step} - Fidelity = {1 - loss}")

        if (1 - loss) > precision:
            print(f"Precision reached, stopping...")
            break

    return {"fidelities": fidelities_total, "parameters": parameters_total}
