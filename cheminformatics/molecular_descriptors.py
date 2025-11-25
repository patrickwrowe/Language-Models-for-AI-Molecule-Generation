from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import numpy as np
from typing import Callable

mol_features_methods: dict[str, Callable] = {
    # Basic
    "MolWt" : Descriptors.MolWt,

    # Lipinski
    "NumHDonors" : Lipinski.NumHDonors,
    "NumHAcceptors" : Lipinski.NumHAcceptors,
    "NumHeteroatoms" : Lipinski.NumHeteroatoms,
    "NumRotatableBonds" : Lipinski.NumRotatableBonds,
    "NOCount" : Lipinski.NOCount,
    "NHOHCount" : Lipinski.NHOHCount,
    "RingCount" : Lipinski.RingCount,

    # Crippen
    "MolLogP": Crippen.MolLogP,
    "MolMR": Crippen.MolMR,
}

def molecular_features_from_smiles_list(smiles_list: list[str], smiles_column: str = "canonical_smiles") -> dict:

    molecular_features = {method_name: [] for method_name, _ in mol_features_methods.items()}
    molecular_features[smiles_column] = smiles_list
    
    for s in smiles_list: 
        mol = Chem.MolFromSmiles(s)
        if mol: 
            for method_name, method in mol_features_methods.items():
                molecular_features[method_name].append(method(mol)) 
        else:
            for method_name, method in mol_features_methods.items():
                molecular_features[method_name].append(np.NaN)

    return molecular_features
