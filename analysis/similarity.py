from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

import umap

def compute_similarity_matrix_2(smiles_list: list[str]):
    # Filter out None values from the input list
    valid_smiles = [smiles for smiles in smiles_list if smiles is not None]
    
    mols = [Chem.MolFromSmiles(smiles) for smiles in valid_smiles]

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [morgan_gen.GetFingerprint(mol) for mol in mols]
    n_mols = len(fps)
    similarities = np.zeros((n_mols, n_mols))

    similarities = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])

    return similarities

def fit_umap(smiles_list: list[str], n_neighbors: int = 15, min_dist: float = 0.1, tanimoto: bool = True, **kwargs):
    
    if tanimoto:
        # Compute the similarity matrix
        similarity_matrix = compute_similarity_matrix_2(smiles_list)
        
        # Apply UMAP to the similarity matrix
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, **kwargs)
        embedding = umap_model.fit_transform(1 - similarity_matrix)
    else:
        # Compute the UMAP embedding directly from the SMILES strings
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, **kwargs)
        embedding = umap_model.fit_transform(smiles_list)

    return embedding, umap_model
       