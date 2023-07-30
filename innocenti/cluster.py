from src.Optimizer import Optimizer
from src.model import QubitNetworkGateModel

from src.analytical_conditions import commuting_generator
from src.QubitNetwork import pauli_product

from src.data_saver import save_log

from src.configs import ToffoliConfig, QFTqubitConfig, FredkinConfig, Weight2ParityZConfig, Weight4ParityZConfig, \
    Weight4ParityXConfig, CxNotConfig

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
    parser.add_argument('--instance', default=0)
    parser.add_argument('--gate', default='qft')
    parser.add_argument('--steps', default=1000)
    args = parser.parse_args()
    precision = 0.999
    max_steps = int(args.steps)
    seed = int(args.instance) * 2 ** 8
    gate = args.gate

    if gate == 'toffoli':
        config = ToffoliConfig()
        target_gate = qutip.Qobj(config.unitary, dims=[[2] * 3, [2] * 3], shape=(8, 8))
        generator = commuting_generator(target_gate, interactions='all')
    elif gate == 'fredkin':
        config = FredkinConfig()
        target_gate = qutip.Qobj(config.unitary, dims=[[2] * 3, [2] * 3], shape=(8, 8))
        generator = commuting_generator(target_gate, interactions='all')
    elif gate == 'cccnot':
        config = CxNotConfig(4)
        target_gate = qutip.Qobj(config.unitary, dims=[[2] * 4, [2] * 4], shape=(16, 16))
        generator = commuting_generator(target_gate, interactions='all')
    elif gate == 'qft':
        config = QFTqubitConfig()()
        target_gate = qutip.Qobj(config.unitary, dims=[[2] * 3, [2] * 3], shape=(8, 8))
        generator = commuting_generator(target_gate, interactions='all')
    elif gate == 'w2pz':
        config = Weight2ParityZConfig()
        target_gate = qutip.Qobj(config.unitary, dims=[[2] * 3, [2] * 3], shape=(8, 8), isunitary=True)
        generator = commuting_generator(target_gate, interactions='all')
    elif gate == 'w4pz':
        config = Weight4ParityZConfig()
        target_gate = qutip.Qobj(config.unitary, dims=[[2] * 5, [2] * 5], shape=(32, 32),
                                 isunitary=True)
        generator = commuting_generator(target_gate, interactions='all')
    elif gate == 'w4px':
        config = Weight4ParityXConfig()
        target_gate = qutip.Qobj(config.unitary, dims=[[2] * 5, [2] * 5], shape=(32, 32),
                                 isunitary=True)
        generator = commuting_generator(target_gate, interactions='all')
    else:
        raise NotImplementedError(f"{gate} not implemented")

    print(f"Running {gate} with seed {seed}")

    save_path = f'./data/{str(config)}/max_steps={max_steps}_precision={precision:1.4f}_seed={seed}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(save_path + "/optimization_data.csv"):
        print("Data already exists, skipping...")
    else:
        print("Running optimization...")
        np.random.seed(seed)
        random.seed(seed)
        initial_parameters = 2 * np.random.rand(len(list(generator.free_symbols))) - 1

        tnet = QubitNetworkGateModel(sympy_expr=generator, initial_values=initial_parameters)

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
            precision=precision
        )


        def unitary_fidelity(unitary1, unitary2):
            return np.abs(np.trace(unitary1.conj().T @ unitary2)) / len(unitary1[0])


        optimizer.run()
        Ufull = tnet.get_current_gate(False)
        true_fid = unitary_fidelity(Ufull, config.unitary)
        print(true_fid)
        print("Run completed, saving data...")
        optimizer.log['fidelities'][optimizer.log['n_epoch']] = true_fid
        save_log(optimizer.log, save_path + "optimization_data.csv")
