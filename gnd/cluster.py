import numpy as np
from src import basis, optimize, data
import argparse
from src.configs import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set parameters')
    parser.add_argument('--instance', default=3)
    parser.add_argument('--gate', default='w4px')
    parser.add_argument('--steps', default=100)
    parser.add_argument('--commute', default=0)
    args = parser.parse_args()
    max_steps = int(args.steps)
    seed = int(args.instance) * 2 ** 8
    gate = args.gate
    commute = bool(int(args.commute))
    precision = 0.999
    if gate == 'toffoli':
        config = ToffoliConfig()
    elif gate == 'qtf':
        config = QFTqubitConfig()
    elif gate == 'fredkin':
        config = FredkinConfig()
    elif gate == 'cccnot':
        config = CxNotConfig(4)
    elif gate == 'ccccnot':
        config = CxNotConfig(5)
    elif gate == 'cccccnot':
        config = CxNotConfig(6)
    elif gate == 'w2pz':
        config = Weight2ParityZConfig()
    elif gate == 'w2px':
        config = Weight2ParityXConfig()
    elif gate == 'w3pz':
        config = Weight3ParityZConfig()
    elif gate == 'w3px':
        config = Weight3ParityXConfig()
    elif gate == 'w4pz':
        config = Weight4ParityZConfig()
    elif gate == 'w4px':
        config = Weight4ParityXConfig()
    elif gate == 'w5pz':
        config = Weight5ParityZConfig()
    elif gate == 'w5px':
        config = Weight5ParityXConfig()
    elif gate == 'w6pz':
        config = Weight6ParityZConfig()
    elif gate == 'w6px':
        config = Weight6ParityXConfig()
    else:
        raise NotImplementedError(f"{args.gate} not implemented")
    config.seed = seed
    config.max_steps = max_steps
    config.commute = commute

    print(f"Running {args.gate} with seed {config.seed}, {'commuting ansatz ' if commute else 'no ansatz '}"
          f"and {max_steps} steps")

    full_basis = basis.construct_full_pauli_basis(config.nqubits)
    projection_basis = basis.construct_two_body_pauli_basis(config.nqubits)

    np.random.seed(seed)

    batch_size = 1000
    while True:

        dat = data.OptimizationData(config, load_data=True)
        batch = dat.samples
        steps_per_batch = min(abs(max_steps - batch * batch_size), batch_size)
        print(f"Performing {steps_per_batch} steps ")
        if dat.exists():
            if dat.fidelities(-1)[-1] >= precision:
                print("Data already exists and precision reached, skipping...")
                break
            elif batch * batch_size + steps_per_batch >= max_steps:
                print("Maximum number of steps reached, stopping...")
                break
            else:
                print(f"Continuing optimization... {batch * batch_size + steps_per_batch}/{max_steps}")
                init_parameters = np.array(eval(dat.parameters(-1)[-1]))
                opt = optimize.Optimizer(config.unitary, full_basis, projection_basis, max_steps=steps_per_batch,
                                         commute=commute,
                                         precision=precision, init_parameters=init_parameters)
                dat = data.OptimizationData(config, optimizers=[opt], load_data=True)
                dat.save_data()
        else:
            print("Running optimization...")
            opt = optimize.Optimizer(config.unitary, full_basis, projection_basis, max_steps=steps_per_batch,
                                     commute=commute,
                                     precision=precision)
            dat = data.OptimizationData(config, optimizers=[opt], load_data=False)
            dat.save_data()
