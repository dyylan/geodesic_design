import numpy as np
from src.basis import PauliBasis
from src.optimizer import optimizer_unitary

from configs import ToffoliConfig, Weight4ParityXConfig
import matplotlib.pyplot as plt

config = Weight4ParityXConfig()

if __name__ == "__main__":
    b = PauliBasis(config.nqubits)
    indices = b.two_body_projection_indices()

    init_parameters = 2 * (np.random.rand(sum(indices)) - 1)
    projected_basis = b.basis[[bool(idx) for idx in indices]]
    data_dict = optimizer_unitary(config.unitary, projected_basis, init_parameters, optimizer='adam')
