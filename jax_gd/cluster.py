import numpy as np
from src.basis import construct_two_body_pauli_basis
from src.optimizer import optimizer_unitary
from src.data_saver import save_log
import argparse
from configs import ToffoliConfig, QFTqubitConfig, FredkinConfig, Weight2ParityZConfig, Weight4ParityZConfig, \
    Weight4ParityXConfig
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set parameters')

    parser.add_argument('--instance', default=1)
    parser.add_argument('--gate', default='toffoli')
    parser.add_argument('--steps', default=1000)
    args = parser.parse_args()
    precision = 0.999
    max_steps = int(args.steps)
    seed = int(args.instance) * 2 ** 8
    gate = args.gate

    if gate == 'toffoli':
        config = ToffoliConfig()
    elif gate == 'qtf':
        config = QFTqubitConfig()
    elif gate == 'fredkin':
        config = FredkinConfig()
    elif gate == 'w2pz':
        config = Weight2ParityZConfig()
    elif gate == 'w4pz':
        config = Weight4ParityZConfig()
    elif gate == 'w4px':
        config = Weight4ParityXConfig()
    else:
        raise NotImplementedError(f"{gate} not implemented")

    save_path = f'./data/{gate}/max_steps={max_steps}_precision={precision:1.4f}_seed={seed}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(save_path + "/optimization_data.csv"):
        print("Data already exists, skipping...")
    else:
        print("Running optimization...")

        np.random.seed(seed)
        projected_basis = construct_two_body_pauli_basis(config.nqubits).basis

        data_dict = optimizer_unitary(config.unitary, projected_basis, optimizer='adam')
        save_log(data_dict, save_path + "/optimization_data.csv")
