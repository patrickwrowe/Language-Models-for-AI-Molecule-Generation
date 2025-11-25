from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

def molecular_weight_from_smiles_list(smiles_list: list[str]):

    molecular_weights = []
    
    for s in smiles_list: 
        mol = Chem.MolFromSmiles(s)
        if mol: 
            molecular_weights.append(Descriptors.MolWt(mol))
        else:
            molecular_weights.append(np.NaN)