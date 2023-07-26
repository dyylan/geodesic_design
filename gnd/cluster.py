import numpy as np
from src import basis, optimize, data
import argparse
from configs import ToffoliConfig, QFTqubitConfig, FredkinConfig, Weight2ParityZConfig, Weight4ParityZConfig, Weight4ParityXConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set parameters')

    parser.add_argument('--instance', default=1)
    parser.add_argument('--gate', default='toffoli')
    parser.add_argument('--steps', default=1000)
    args = parser.parse_args()
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
        raise NotImplementedError(f"{args.gate} not implemented")
    config.seed = seed
    config.max_steps = max_steps

    print(f"Running {args.gate} with seed {config.seed} and {max_steps} steps")

    b = basis.PauliBasis(config.nqubits)

    init_parameters = 2 * np.random.rand(len(b.basis)) - 1
    init_parameters = b.linear_span(init_parameters)
    dat = data.OptimizationData(config, load_data=False)
    if dat.exists():
        print("Data already exists, skipping...")
    else:
        print("Running optimization...")
        optimize = optimize.Optimizer(config.unitary, b, init_parameters, max_steps=max_steps)
        dat = data.OptimizationData(config, optimizers=[optimize], load_data=True)
        dat.save_data()
