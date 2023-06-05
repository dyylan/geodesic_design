# Unitary gate design with time-independent Hamiltonians

## GND package
The package is in the folder `gnd`, see `main.py` for an example. 

Target gates are defined as classes in the `configs.py`. The `__str__` method returns the name 
of the folder saved under `data` and `__dir__` gives the unique variables in the file name. The config
also contains information such as the number of qubits. 

The Pauli word basis is given by `gnd.basis.PauliBasis` with the number of qubits. Parameters can be 
projected onto the one- and two-body projection with the basis method `.two_body_projection(parameters)`.

The optimizer is called from `gnd.optimize`, and requires a target unitary, basis, and some initial parameters.

The data from the optimizer is then stored in a `gnd.data.OptimizationData` class, which handles the saving and loading of data.
The data stored is the "steps", "parameters", "fidelities", and "step_sizes".
The data class takes a list of optimizers and stores them as a CSV file in the correct folder defined by the config.
Data can also be loaded using the data class and the correct config.

## Benchmarks

As comparison we provide two benchmarks,

1. First, we compare the with the work by Innocenti (2020):
> Luca Innocenti, Leonardo Banchi, Alessandro Ferraro, Sougato Bose, Mauro Paternostro (2020). "*Supervised learning of time-independent Hamiltonians for gate design*". [*New J. Phys.* **22** 065001](https://doi.org/10.1088/1367-2630/ab8aaf) ([arXiv:1803.07119](https://arxiv.org/abs/1803.07119)).

who kindly provided code at https://github.com/lucainnocenti/quantum-gate-learning-1803.07119 that we adapted for ou purposes.
See the folder `innocenti` for details.

2. In addition to comparing with a supervised learning method, we compare with a naive gradient descent method in the Lie algebra.
See the folder `jax_gd` for details.
