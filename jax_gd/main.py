from src.basis import construct_two_body_pauli_basis
from src.optimizer import optimizer_unitary

from src.configs import ToffoliConfig, Weight4ParityXConfig

config = ToffoliConfig()

if __name__ == "__main__":
    projected_basis = construct_two_body_pauli_basis(config.nqubits).basis

    data_dict = optimizer_unitary(config.unitary, projected_basis, optimizer='adam')
