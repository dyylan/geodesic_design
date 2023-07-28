import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
from src import basis, optimize, data, utils, lie
import jax
from src.configs import ToffoliConfig, QFTqubitConfig, Weight2ParityZConfig, Weight3ParityZConfig, Weight4ParityZConfig, \
    CnotConfig, FredkinConfig, CxNotConfig
# config = QFTqubitConfig()
# config = ToffoliConfig()
# config = CnotConfig()
# config = Weight2ParityZConfig()
# config = Weight3ParityZConfig()
# config = Weight4ParityZConfig()
# config = FredkinConfig()
config = CxNotConfig(4)

if __name__ == "__main__":
    full_basis = basis.construct_full_pauli_basis(config.nqubits)
    projection_basis = basis.construct_two_body_pauli_basis(config.nqubits)
    # init_parameters = utils.prepare_random_initial_parameters(config.unitary, b)
    # ham = -1.j * spla.logm(config.unitary)
    # target_params = lie.Hamiltonian.parameters_from_hamiltonian(ham, b)
    # target_ham = lie.Hamiltonian(b, target_params)
    # init_ham = lie.Hamiltonian(b, init_parameters)
    # print(init_ham.matrix @ target_ham.matrix - target_ham.matrix @ init_ham.matrix) # Check the commutator is zero

    # optimize = optimize.Optimizer(config.unitary, b, init_parameters, max_steps=3000)
    # seeds = [1] + [n*256 for n in range(1,101,1)]

    # num_steps = []
    # num_steps2 = []
    # for seed in seeds:
    #     print(f"Collecting data for seed: {seed}", end="\r")
    #     config.seed = seed
    #     config2.seed = seed
    #     dat = data.OptimizationData(config, optimizers=[], load_data=True)
    #     if dat.samples > 0:
    #         num_steps.append(dat.steps(1)[-1])
    #     dat2 = data.OptimizationData(config2, optimizers=[], load_data=True)
    #     if dat2.samples > 0:
    #         num_steps2.append(dat2.steps(1)[-1])

    # plt.hist(num_steps, bins=[i+1 for i in range(80)], alpha=0.5, label="Toffoli")
    # plt.hist(num_steps2, bins=[i+1 for i in range(80)], alpha=0.5, label="Weight-2 Z parity")
    # plt.legend()
    # plt.show()

    # init_parameters = b.two_body_projection(init_parameters)
    # dat = data.OptimizationData(config, load_data=False)
    import time
    start = time.time()
    optimize = optimize.Optimizer(config.unitary, full_basis, projection_basis, max_steps=1000, max_step_size=2,
                                  commute=True)
    print(f"elapsed time = {time.time()-start}")
    # dat = data.OptimizationData(config, optimizers=[optimize], load_data=True)
    # # dat.save_data()
    # full_basis = basis.construct_full_pauli_basis(config.nqubits)
    # projection_basis = basis.construct_two_body_pauli_basis(config.nqubits)
    #
    #
    # # init_parameters = utils.prepare_random_initial_parameters(config.unitary, b)
    # # ham = -1.j * spla.logm(config.unitary)
    # # target_params = lie.Hamiltonian.parameters_from_hamiltonian(ham, b)
    # # target_ham = lie.Hamiltonian(b, target_params)
    # # init_ham = lie.Hamiltonian(b, init_parameters)
    # # print(init_ham.matrix @ target_ham.matrix - target_ham.matrix @ init_ham.matrix) # Check the commutator is zero
    #
    # # optimize = optimize.Optimizer(config.unitary, b, init_parameters, max_steps=3000)
    # # seeds = [1] + [n*256 for n in range(1,101,1)]
    #
    # # num_steps = []
    # # num_steps2 = []
    # # for seed in seeds:
    # #     print(f"Collecting data for seed: {seed}", end="\r")
    # #     config.seed = seed
    # #     config2.seed = seed
    # #     dat = data.OptimizationData(config, optimizers=[], load_data=True)
    # #     if dat.samples > 0:
    # #         num_steps.append(dat.steps(1)[-1])
    # #     dat2 = data.OptimizationData(config2, optimizers=[], load_data=True)
    # #     if dat2.samples > 0:
    # #         num_steps2.append(dat2.steps(1)[-1])
    #
    # # plt.hist(num_steps, bins=[i+1 for i in range(80)], alpha=0.5, label="Toffoli")
    # # plt.hist(num_steps2, bins=[i+1 for i in range(80)], alpha=0.5, label="Weight-2 Z parity")
    # # plt.legend()
    # # plt.show()
    #
    # # init_parameters = b.two_body_projection(init_parameters)
    # # dat = data.OptimizationData(config, load_data=False)
    # np.random.seed(1234)
    # optimize = optimize.Optimizer(config.unitary, full_basis, projection_basis, max_steps=1000, max_step_size=2)
    # # dat = data.OptimizationData(config, optimizers=[optimize], load_data=True)
    # # dat.save_data()