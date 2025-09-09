import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.Chem import Descriptors

from fingerprints import assign_fp


class SimilarityMetrics:
    @staticmethod
    def bitvect(fp_array):
        bitvect = ExplicitBitVect(len(fp_array))
        for i, bit in enumerate(fp_array):
            if bit:
                bitvect.SetBit(i)
        return bitvect

    @staticmethod
    def distribution(fps, metric):
        metrics = {
            'Tanimoto': DataStructs.TanimotoSimilarity,
            'Dice': DataStructs.DiceSimilarity,
            'Asymmetric': DataStructs.AsymmetricSimilarity
            }
        similarities = []
        similarity_fn = metrics[metric]
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = similarity_fn(fps[i], fps[j])
                similarities.append(sim)

        return similarities


class SimilarityPlot:
    def __init__(self, smiles_list):
        self.smiles_list  = smiles_list
        self.fingerprints = self._generate_fingerprints()

    def _generate_fingerprints(self):
        fingerprints, _ = assign_fp(self.smiles_list)
        return [SimilarityMetrics.bitvect(
            fp) for fp in fingerprints
            ]

    def plot(
        self, metrics=None, 
        out_path=None, 
        bw_adjust=1.6):

        if metrics is None:
            metrics = [
            'Tanimoto', 'Dice', 'Asymmetric'
            ]
        similarity_data = {
            m: SimilarityMetrics.distribution(
                self.fingerprints, m
                )
            for m in metrics
        }
        plt.figure(figsize=(5, 3))
        palette = ['#4E79A7', '#F28E2B', '#E15759']
        for metric, color in zip(metrics, palette):
            sns.kdeplot(
                similarity_data[metric],
                label=metric,
                color=color,
                linewidth=1,
                fill=True,
                alpha=0.6,
                bw_adjust=bw_adjust
            )
        plt.xlabel("Similarity Score", fontsize=16)
        plt.ylabel("Density", fontsize=16)
        plt.legend(title='Metrics', 
            title_fontsize=13, fontsize=11
            )
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=300)
        plt.show()



class SimilarityAnalysis:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.out_path = "../output/figures/Similarity.png"

    def run(self):
        smiles_list = self.dataframe['SMILES'].tolist()
        similarity_plotter = SimilarityPlot(smiles_list)
        similarity_plotter.plot(
            out_path=self.out_path
            )


class MolecularProperties:
    property_names = [
        'cLogP', 'MW', 'Rotatable bonds', 'tPSA'
        ]
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol     = Chem.MolFromSmiles(smiles)

    def calculate_properties(self):
        if self.mol:
            logp = Descriptors.MolLogP(self.mol)
            mw = Descriptors.MolWt(self.mol)
            rotn = Descriptors.NumRotatableBonds(self.mol)
            tpsa = Descriptors.TPSA(self.mol)
            return pd.Series([logp, mw, rotn, tpsa], 
                index=self.property_names
                )
        
        return pd.Series([None] * len(self.property_names), 
                index=self.property_names
                )


class ScatterPlotMatrix:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def add_properties(self):
        self.dataframe.loc[:, 
            MolecularProperties.property_names] = (
            self.dataframe['SMILES']
                .apply(lambda sm: MolecularProperties(
                    sm).calculate_properties())
        )

    def _trim_outliers(self, df, lower_q=0.01, upper_q=0.99):
        props  = MolecularProperties.property_names
        qs     = df[props].quantile([lower_q, upper_q])
        low, high = qs.loc[lower_q], qs.loc[upper_q]
        mask   = ((df[props] >= low) & (
            df[props] <= high)).all(axis=1)
        return df[mask]

    def plot(self):
        out_path = "../output/figures/Scatterplot.png"
        props    = MolecularProperties.property_names
        df = (self.dataframe
            .dropna(subset=props + ['Class'])
            .assign(ClassLabel=lambda d: d['Class']
                .map({1:'Active', 0:'Inactive'}))
        )
        df = self._trim_outliers(df)
        sns.set_theme(style="white")
        pairplot = sns.pairplot(
            df,
            vars=props,
            hue='ClassLabel',
            palette={'Active':'#2AAD0F','Inactive':'#A1C3CE'},
            diag_kind='kde',
            diag_kws={
                'fill': True,
                'common_norm': False,
                'alpha': 0.4,
                'linewidth': 1.0},
            plot_kws={
                'alpha':0.8,
                's':25,
                'edgecolor':'k'},
            height=3,
            aspect=1
            )
        pairplot.fig.set_size_inches(12, 8)
        for ax in pairplot.axes.flatten():
            for spine in ['top','bottom','left','right']:
                ax.spines[spine].set_visible(True)
            ax.tick_params(axis='both', direction='in', length=4)
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
        pairplot._legend.remove()
        pairplot.figure.tight_layout()
        pairplot.savefig(out_path)
        plt.show()