import pandas as pd
import numpy as np


def save_log(log, save_path):
    data_dict = {
        "sample": np.ones(log['n_epoch'], dtype=int),
        "steps": np.arange(0, log['n_epoch']),
        "parameters": [list(p) for p in log['parameters'][:log['n_epoch']]],
        "fidelities": log['fidelities'][:log['n_epoch']],
        "step_sizes": np.zeros(log['n_epoch']),
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(save_path, index=False)
