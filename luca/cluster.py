from qubit_network.Optimizer import Optimizer
from qubit_network.model import QubitNetworkGateModel

from qubit_network.analytical_conditions import commuting_generator
from qubit_network.QubitNetwork import pauli_product

from data_saver import save_log

import qutip
import sympy
import numpy as np
import argparse
import os
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def J(*args):
    return sympy.Symbol('J' + ''.join(str(arg) for arg in args))


def pauli(*args):
    return pauli_product(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set parameters')
    parser.add_argument('--instance', default=1)
    parser.add_argument('--gate', default='toffoli')
    args = parser.parse_args()
    precision = 0.999
    max_steps = 1000
    seed = int(args.instance) * 2 ** 8
    gate = args.gate

    if gate == 'toffoli':
        generator = J(3, 0, 0) * pauli(3, 0, 0)
        generator += J(0, 3, 0) * pauli(0, 3, 0)
        generator += J(0, 0, 1) * pauli(0, 0, 1)
        generator += J(3, 0, 3) * (pauli(0, 0, 3) + pauli(3, 0, 3))
        generator += J(0, 3, 3) * (pauli(0, 0, 3) + pauli(0, 3, 3))
        generator += (J(1, 0, 1) * pauli(1, 0, 0) + J(0, 1, 1) * pauli(0, 1, 0)) * (
                pauli(0, 0, 0) + pauli(0, 0, 1))
        generator += J(2, 2, 0) * (pauli(1, 1, 0) + pauli(2, 2, 0))
        generator += J(3, 3, 0) * pauli(3, 3, 0)
        target_gate = qutip.toffoli()
    elif gate == 'fredkin':
        generator = commuting_generator(qutip.fredkin(), interactions='diagonal')
        target_gate = qutip.fredkin()
    else:
        raise NotImplementedError(f"{gate} not implemented")

    print(f"Running {gate} with seed {seed}")

    # import pickle
    # with open('file.c', 'rb') as file:
    #     log = pickle.load(file)

    save_path = f'./data/{gate}/max_steps={max_steps}_precision={precision:1.4f}_seed={seed}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(save_path + "/optimization_data.csv"):
        print("Data already exists, skipping...")
    else:
        print("Running optimization...")
        np.random.seed(seed)
        random.seed(seed)
        tnet = QubitNetworkGateModel(sympy_expr=generator, initial_values=1)
        optimizer = Optimizer(
            net=tnet,
            learning_rate=1,
            decay_rate=.005,
            n_epochs=max_steps,
            batch_size=2,
            target_gate=target_gate,
            training_dataset_size=200,
            test_dataset_size=100,
            sgd_method='momentum',
            precision=0.999
        )
        optimizer.run()
        print("Run completed, saving data...")
        save_log(optimizer.log, save_path + "optimization_data.csv")