import numpy as np
from src.basis import PauliBasis
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
    args = parser.parse_args()
    gate = args.gate
    max_steps = 1000
    precision = 0.999
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

        b = PauliBasis(config.nqubits)
        indices = b.two_body_projection_indices()
        np.random.seed(seed)
        init_parameters = 2 * (np.random.rand(sum(indices)) - 1)
        projected_basis = b.basis[[bool(idx) for idx in indices]]

        data_dict = optimizer_unitary(config.unitary, projected_basis, init_parameters, optimizer='adam')
        save_log(data_dict, save_path + "/optimization_data.csv")
