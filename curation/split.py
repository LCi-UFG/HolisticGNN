import os
import random
import math
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class DataSplitter:
    def __init__(
        self, df, 
        out_path, 
        train_size=0.8, 
        val_size=0.1, 
        random_state=None):

        self.df = df
        self.out_path = out_path
        self.train_size = train_size
        self.val_size = val_size
        self.random_state = random_state
        self._prepare_out_path()

    def _prepare_out_path(self):
        os.makedirs(self.out_path, exist_ok=True)

    def save_splits(
        self, train, val, test):
        train.to_csv(os.path.join(
            self.out_path, 
            'train.csv'), index=False
            )
        val.to_csv(os.path.join(
            self.out_path, 
            'val.csv'), index=False
            )
        test.to_csv(os.path.join(
            self.out_path, 
            'test.csv'), index=False
            )
        print(f"Training set size: {train.shape}")
        print(f"Validation set size: {val.shape}")
        print(f"Test set size: {test.shape}")


class RandomSplitter(DataSplitter):
    def __init__(
        self, df, 
        out_path, 
        train_size=0.8, 
        val_size=0.1, 
        random_state=None):

        super().__init__(
            df, out_path, 
            train_size, 
            val_size, 
            random_state)

    def split(self):
        total_size = len(self.df)
        train_size = math.floor(
            self.train_size * total_size
            )
        val_size = math.floor(
            self.val_size * total_size
            )
        shuffled_indices = self.df.sample(
            frac=1, random_state=self.random_state).index

        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[
            train_size:train_size + val_size
            ]
        test_indices = shuffled_indices[
            train_size + val_size:
            ]
        train = self.df.loc[train_indices]
        val = self.df.loc[val_indices]
        test = self.df.loc[test_indices]

        return train, val, test

    def execute(self):
        train, val, test = self.split()
        self.save_splits(train, val, test)


class ScaffoldSplitter(DataSplitter):
    @staticmethod
    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)
                )
            return Chem.MolToSmiles(scaffold)
        return None

    def apply_scaffold(self):
        self.df['scaffold'] = self.df[
            'SMILES'].apply(self.get_scaffold)
        self.df = self.df.loc[self.df[
            'scaffold'].notna()]

    def split(self):
        scaffold_groups = self.df.groupby('scaffold')
        scaffold_dict = {scaffold: group.index.tolist
            () for scaffold, group in scaffold_groups
            }
        train_idx, val_idx, test_idx = [], [], []
        scaffolds = list(scaffold_dict.keys())
        random.shuffle(scaffolds)

        total_size = len(self.df)
        train_size = int(
            self.train_size * total_size
            )
        val_size = int(
            self.val_size * total_size
            )
        test_size = total_size - train_size - val_size

        for scaffold in scaffolds:
            scaffold_indices = scaffold_dict[scaffold]
            
            if len(train_idx) + len(
                scaffold_indices) <= train_size:
                train_idx.extend(scaffold_indices)
            
            elif len(val_idx) + len(
                scaffold_indices) <= val_size:
                val_idx.extend(scaffold_indices)
            else:
                test_idx.extend(scaffold_indices)

        train_idx, val_idx, test_idx = self.balance_splits(
            train_idx, 
            val_idx, 
            test_idx, 
            train_size, 
            val_size, 
            test_size
            )
        train = self.df.loc[
            train_idx].drop(columns=['scaffold'])
        val = self.df.loc[
            val_idx].drop(columns=['scaffold'])
        test = self.df.loc[
            test_idx].drop(columns=['scaffold'])
        return train, val, test

    def balance_splits(
        self, 
        train_idx, 
        val_idx, 
        test_idx, 
        train_size, 
        val_size, 
        test_size):

        while len(train_idx) > train_size:
            idx_to_move = train_idx.pop()
            if len(val_idx) < val_size:
                val_idx.append(idx_to_move)
            else:
                test_idx.append(idx_to_move)
        while len(val_idx) > val_size:
            idx_to_move = val_idx.pop()
            test_idx.append(idx_to_move)
        while len(test_idx) > test_size:
            idx_to_move = test_idx.pop()
            val_idx.append(idx_to_move)
        return train_idx, val_idx, test_idx

    def execute(self):
        self.apply_scaffold()
        train, val, test = self.split()
        self.save_splits(train, val, test)


def DataSpliter(
    method, 
    df, 
    out_path):

    if method == "random":
        splitter = RandomSplitter(df, out_path)
    elif method == "scaffold":
        splitter = ScaffoldSplitter(df, out_path)
    else:
        raise ValueError(
            f"Invalid split method: {method}"
        )
    splitter.execute()