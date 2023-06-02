import numpy as np
from src import basis, optimize, data
import argparse
from configs import ToffoliConfig, QFTqubitConfig, FredkinConfig, Weight2ParityZConfig, Weight4ParityZConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set parameters')

    parser.add_argument('--instance', default=1)
    parser.add_argument('--gate', default='toffoli')
    args = parser.parse_args()
    if args.gate == 'toffoli':
        config = ToffoliConfig()
    elif args.gate == 'qtf':
        config = QFTqubitConfig()
    elif args.gate == 'fredkin':
        config = FredkinConfig()
    elif args.gate == 'w2pz':
        config = Weight2ParityZConfig()
    elif args.gate == 'w4pz':
        config = Weight4ParityZConfig()
    elif args.gate == 'w4px':
        config = Weight4ParityZConfig()
    else:
        raise NotImplementedError(f"{args.gate} not implemented")

    config.seed = int(args.instance) * 2 ** 8
    print(f"Running {args.gate} with seed {config.seed}")
    import os
    print(os.path.exists("data/toffoli/max_steps=1000_precision=0.9990_seed=256/optimization_data.csv"))
    b = basis.PauliBasis(config.nqubits)

    init_parameters = 2 * np.random.rand(len(b.basis)) - 1
    init_parameters = b.two_body_projection(init_parameters)
    dat = data.OptimizationData(config, load_data=False)
    if dat.exists():
        print("Data already exists, skipping...")
    else:
        print("Running optimization...")
        optimize = optimize.Optimizer(config.unitary, b, init_parameters, max_steps=1000)
        dat = data.OptimizationData(config, optimizers=[optimize], load_data=True)
        dat.save_data()
