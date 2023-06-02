import pandas as pd
import numpy as np


def save_log(log, save_path):
    data_dict = {
        "sample": np.ones(len(log['fidelities']), dtype=int),
        "steps": np.arange(0, len(log['fidelities'])),
        "parameters": [list(p) for p in log['parameters']],
        "fidelities": np.array(log['fidelities']),
        "step_sizes": np.zeros(len(log['fidelities'])),
    }

    df = pd.DataFrame(data_dict)
    df.to_csv(save_path, index=False)
