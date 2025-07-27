
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os
import datetime

def extract_training_losses(metadata: dict) -> dict:
    """
    Extract the training and validation losses from the metadata dictionary. Moves any torch tensors to CPU.
    """

    train_losses = []
    avg_train_losses = []
    val_losses = []
    avg_val_losses = []

    for epoch in metadata["training_epochs"]:
        # Move to CPU if tensor, else leave as is
        train_loss = epoch["train_loss"]
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.cpu()
        elif isinstance(train_loss, list):
            train_loss = [tl.cpu() if isinstance(tl, torch.Tensor) else tl for tl in train_loss]
        train_losses.append(train_loss)

        avg_train_loss = epoch["avg_train_loss"]
        if isinstance(avg_train_loss, torch.Tensor):
            avg_train_loss = avg_train_loss.cpu()
        avg_train_losses.append(avg_train_loss)

        val_loss = epoch["val_loss"]
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.cpu()
        elif isinstance(val_loss, list):
            val_loss = [vl.cpu() if isinstance(vl, torch.Tensor) else vl for vl in val_loss]
        val_losses.append(val_loss)

        avg_val_loss = epoch["avg_val_loss"]
        if isinstance(avg_val_loss, torch.Tensor):
            avg_val_loss = avg_val_loss.cpu()
        avg_val_losses.append(avg_val_loss)

    # return as dict of arrays
    import numpy as np
    return {
        "train_losses": np.array(train_losses),
        "avg_train_losses": np.array(avg_train_losses),
        "val_losses": np.array(val_losses),
        "avg_val_losses": np.array(avg_val_losses),
    }


def validate_smiles_strings(smiles_list: list[str]) -> list[bool]:
    """
    Validate a list of SMILES strings and return a list of booleans indicating validity.
    """
    return [validate_smiles_string(smiles) for smiles in smiles_list]

def validate_smiles_string(smiles: str):
    
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is not None: 
        return True 
    else: 
        return False

def draw_molecule(smiles: str):
    if not validate_smiles_string(smiles):
        raise RuntimeError("Invalid SMILES string, could not be parsed.")
    else:
        mol = Chem.MolFromSmiles(smiles)
        return Draw.MolToImage(mol)

def save_model_weights(prefix, model, data):
    if not os.path.exists(os.path.join("../", "models")):
        os.makedirs(os.path.join("../", "models"))
        
    save_name = prefix + model.__class__.__name__ \
                + "-" + data.__class__.__name__ \
                + "-" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') \
                + ".pt"
    print(f"Saving model to {os.path.join('../', 'models', save_name)}")
    model.save_weights(os.path.join("../", "models", save_name))
    