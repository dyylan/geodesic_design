# Unitary gate design with time-independent Hamiltonians

The package is in `gnd`, see `main.py` for an example. 

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

