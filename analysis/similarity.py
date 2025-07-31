from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

# def compute_similarity_matrix(smiles_list: list[str]):
#     mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

#     print("plink")

#     if None in mols:
#         print("Warning, None found in Mols. Probably due to invalid SMILES. Skipping")
#         mols = [mol for mol in mols if mol]
    
#     fps = [FingerprintMols.FingerprintMol(mol) for mol in mols]

#     similarity_matrix = []
#     for fp in fps:
#         similarities = DataStructs.BulkTanimotoSimilarity(fp, fps)
#         similarity_matrix.append(similarities)

#     return similarity_matrix

#     # return DataStructs.GetTanimotoSimMat(
#     #     fps
#     # )



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
