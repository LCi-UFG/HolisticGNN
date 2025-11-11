import numpy as np
import pandas as pd
from tabulate import tabulate
from rdkit import Chem


def load_data(file_path):
    df = pd.read_csv(
        file_path, 
        sep=None, 
        engine="python"
    )
    smiles = df.iloc[:, 1].values
    targets = df.iloc[:, 2:].values
    
    return smiles, targets


def to_float_array(arr):
    def convert(val):
        if isinstance(val, float):
            return val
        elif isinstance(val, str):
            val = val.replace(',', '.')
            try:
                return float(val)
            except ValueError:
                return np.nan
        else:
            return np.nan
    return np.vectorize(convert)(arr)