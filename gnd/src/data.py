import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class OptimizationData:
    """
    Data handling object for saving, loading, and plotting optimizer data
    
    Parameters
    ----------
    target_unitary : optimize.Optimizer
        Pre-initialised Optimizer object

    Attributes
    ----------
    config : Config
        Configuration class object that determines how to save and load optimisation data
    folder : Str, default="data"
        Main folder to save the data
    extension : Str, default="csv"
        File extension of the saved data
    """

    def __init__(self, config, optimizers=[], load_data=True, folder="data", extension="csv"):
        self.config = config
        self.folder = folder
        self.extension = extension
        self.samples = 0
        self.optimizers = []
        if load_data:
            self._load_optimization_data()
        for optimizer in optimizers:
            self.add_optimizer(optimizer)

    def steps(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]["steps"]

    def parameters(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]["parameters"]

    def fidelities(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]["fidelities"]

    def running_fidelities(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        fs = self.optimizers[index]["fidelities"]
        running_fidelities = [fs[0]]
        for i, f in enumerate(fs[1:]):
            fr = f if f > running_fidelities[i - 1] else running_fidelities[i - 1]
            running_fidelities.append(fr)
        return running_fidelities

    def step_sizes(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return self.optimizers[index]["step_sizes"]

    def max_fidelity(self, sample):
        self._is_optimizer_loaded()
        index = self._find_sample(sample)
        return max(self.optimizers[index]["fidelities"])

    def add_optimizer(self, optimizer):
        self.samples += 1
        self.optimizers.append(self._construct_data_dict(optimizer))

    def save_data(self):
        self._is_optimizer_loaded()
        filepath = self._generate_filepath()
        dfs = {}
        for i, optimizer in enumerate(self.optimizers):
            dfs[i + 1] = pd.DataFrame(optimizer)
        concatdfs = pd.concat(dfs)
        concatdfs.to_csv(filepath, index=False)
        return dfs

    def exists(self):
        filepath = self._generate_filepath()
        return os.path.exists(filepath)

    def plot_parameters(self, basis, sample, title=False, figsize=[14, 6]):
        labels = ["".join(map(str, l)) for l in basis.labels]
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(labels, self.parameters(sample)[-1])
        ax.grid()
        if title:
            plt.title(title)
        plt.xticks(rotation=90)
        plt.show()

    def plot_fidelities(self, title=False, figsize=[12, 6]):
        fig, ax = plt.subplots(figsize=figsize)
        for s in range(1, self.samples + 1):
            ax.plot(self.steps(s), self.fidelities(s))
        if title:
            plt.title(title)
        plt.show()

    def plot_step_sizes(self, title=False, figsize=[12, 6]):
        fig, ax = plt.subplots(figsize=figsize)
        for s in range(1, self.samples + 1):
            ax.plot(self.steps(s), self.step_sizes(s))
        if title:
            plt.title(title)
        plt.show()

    def _load_optimization_data(self):
        filepath = self._generate_filepath()
        if not os.path.isfile(filepath):
            return 0
        try:
            dfs = pd.read_csv(filepath,
                              index_col=False,
                              )
        except pd.errors.ParserError:
            print("Last batch failed, dropping last batch")
            with open(filepath, 'r+') as file:
                # Move the pointer (similar to a cursor in a text editor) to the end of the file
                file.seek(0, os.SEEK_END)
                # This code means the following code skips the very last character in the file -
                # i.e. in the case the last line is null we delete the last line
                # and the penultimate one
                pos = file.tell() - 1

                # Read each character in the file one at a time from the penultimate
                # character going backwards, searching for a newline character
                # If we find a new line, exit the search
                while pos > 0 and file.read(1) != "\n":
                    pos -= 1
                    file.seek(pos, os.SEEK_SET)
                # So long as we're not at the start of the file, delete all the characters ahead
                # of this position
                if pos > 0:
                    file.seek(pos, os.SEEK_SET)
                    file.truncate()

            dfs = pd.read_csv(filepath,
                              index_col=False,
                              skiprows=[-1])
            samples = dfs["sample"].max()
            dfs.drop(dfs[dfs["sample"] == samples].index, inplace=True)
            dfs.to_csv(filepath, index=False)
        dfs = pd.read_csv(filepath,
                          index_col=False,
                          )
        dfs.dropna(axis=0, inplace=True)
        samples = dfs["sample"].max()
        self.samples += samples
        for sample in range(1, samples + 1):
            df = dfs[dfs["sample"] == sample]
            self.optimizers.append(df.to_dict("list"))
        return self.optimizers

    def _generate_filepath(self, name="optimization_data", extension="csv", float_precision=4):
        config_attributes = dir(self.config)
        conf_folder = ""
        for attr in config_attributes:
            a = getattr(self.config, attr)
            a = int(a) if np.isclose(int(a), a) else float(a)
            if type(a) is int:
                a_str = f"m{abs(a)}" if str(a)[0] == "-" else str(a)
                conf_folder += "_" + attr + "=" + a_str
            elif type(a) is float:
                a_str = f"m{abs(a):.{float_precision}f}" if str(a)[0] == "-" else f"{a:.{float_precision}f}"
                conf_folder += "_" + attr + "=" + a_str
            elif type(a) is bool:
                a_str = f"m{a}"
                conf_folder += "_" + attr + "=" + a_str
        conf_folder = conf_folder[1:]
        root_folder = f"{self.folder}/{str(self.config)}/{conf_folder}"
        directory = os.getcwd() + "/" + root_folder
        if not os.path.exists(directory):
            os.makedirs(directory)
        return root_folder + f"/{name}.{extension}"

    def _construct_data_dict(self, optimizer):
        data_dict = {
            "sample": [self.samples] * (optimizer.steps[-1] + 1),
            "steps": optimizer.steps,
            "parameters": [list(p) for p in optimizer.parameters],
            "fidelities": optimizer.fidelities,
            "step_sizes": optimizer.step_sizes,
        }
        return data_dict

    def _is_optimizer_loaded(self):
        if self.optimizers is []:
            raise ValueError("Must use add_optimizer before you can check the parameters.")

    def _find_sample(self, sample):
        for i, dic in enumerate(self.optimizers):
            if sample in dic["sample"]:
                return i
        return -1
