import numpy as np
from gnd import basis, optimize, data

from configs import ToffoliConfig, QFTqubitConfig, Weight2ParityZConfig, Weight4ParityZConfig


config = Weight4ParityZConfig()

if __name__ == "__main__":
    b = basis.PauliBasis(config.nqubits)

    init_parameters = 2*np.random.rand(len(b.basis))-1

    init_parameters = b.two_body_projection(init_parameters)

    optimize = optimize.Optimizer(config.unitary, b, init_parameters, max_steps=1000)
    dat = data.OptimizationData(config, optimizers=[optimize], load_data=True)
    dat.save_data()

    dat.plot_fidelities()
    dat.plot_step_sizes()
    dat.plot_parameters(b, 3)