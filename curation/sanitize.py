from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

from rules import ATOMIC_NUMBER, METALS_LIST

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def remove_metals(mol):
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in METALS_LIST:
            mol.RemoveAtom(atom.GetIdx())
    return mol.GetMol()


def standardize(mol):
    normalizer = rdMolStandardize.Normalizer()
    mol = normalizer.normalize(mol)
    try:
        Chem.Kekulize(
            mol, clearAromaticFlags=True
            )
    except Chem.KekulizeException:
        pass
    reionizer = rdMolStandardize.Reionizer()
    mol = reionizer.reionize(mol)
    
    return mol


def ChemUtils(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        remover = SaltRemover(defnData="[!#1]")
        mol = remover.StripMol(
            mol, dontRemoveEverything=False)
        mol = remove_metals(mol)
    
        try:
            Chem.Kekulize(
                mol, clearAromaticFlags=True)
        except Chem.KekulizeException:
            pass
        mol = standardize(mol)
        allowed = set(ATOMIC_NUMBER())
        if any(atom.GetAtomicNum() not in allowed
                for atom in mol.GetAtoms()):
            return None
        Chem.SanitizeMol(mol)
        
        return Chem.MolToSmiles(mol)
    
    except Exception:
        return None


def trustBTverify(df, smiles='SMILES'):
    total = len(df)
    sanitized = df[
        smiles].apply(
        lambda smi: ChemUtils(smi)
        if ChemUtils(smi) else None
        )
    removed = (total - sanitized.notna().sum())
    
    print(f"Total: {total}")
    print(f"Removed: {removed}")
    print(f"Remaining: {total - removed}")
    
    df = df[sanitized.notna()]
    df.loc[:, smiles] = sanitized.dropna().values

    return df