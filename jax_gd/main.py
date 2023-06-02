import numpy as np
import matplotlib.pyplot as plt
from gnd import basis, optimize, data

from configs import ToffoliConfig, QFTqubitConfig, Weight2ParityZConfig, Weight4ParityZConfig


config = ToffoliConfig()
config2 = Weight2ParityZConfig()

if __name__ == "__main__":
    b = basis.PauliBasis(config.nqubits)

    # init_parameters = 2*np.random.rand(len(b.basis))-1
    # init_parameters = b.two_body_projection(init_parameters)

    # optimize = optimize.Optimizer(config.unitary, b, init_parameters, max_steps=3000)
    seeds = [1] + [n*256 for n in range(1,101,1)] 

    num_steps = []
    num_steps2 = []
    for seed in seeds:
        print(f"Collecting data for seed: {seed}", end="\r")
        config.seed = seed        
        config2.seed = seed
        dat = data.OptimizationData(config, optimizers=[], load_data=True)
        if dat.samples > 0:
            num_steps.append(dat.steps(1)[-1])
        dat2 = data.OptimizationData(config2, optimizers=[], load_data=True)
        if dat2.samples > 0:
            num_steps2.append(dat2.steps(1)[-1])

    plt.hist(num_steps, bins=[i+1 for i in range(80)], alpha=0.5, label="Toffoli")
    plt.hist(num_steps2, bins=[i+1 for i in range(80)], alpha=0.5, label="Weight-2 Z parity")
    plt.legend()
    plt.show()
