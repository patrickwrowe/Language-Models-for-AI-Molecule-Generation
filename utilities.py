import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol
import os
import datetime
from typing import Optional


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

def draw_molecules_as_grid_from_smiles(canonical_smiles: list[str], names: Optional[list[str]], **kwargs):
    """
    Returns an image file with drawn representations of a list of SMILES strings in a grid format. 
    """

    if names: assert len(canonical_smiles) == len(names), f"Num smiles and names must match but got {len(canonical_smiles)} and {len(names)}"

    # Determine number of rows/columns for grid of images.
    root_num_smiles = np.sqrt(len(canonical_smiles))
    if not np.isclose(root_num_smiles % 1, 0):
        cols, rows = int(root_num_smiles), int(root_num_smiles) + 1
        if rows + cols < len(canonical_smiles): rows += 1
    else:
        rows, cols = int(root_num_smiles), int(root_num_smiles)

    print(f"{rows=}, {cols=}")

    # Get mols from smiles strings
    mols = []
    for i, smiles in enumerate(canonical_smiles):
        mols.append(Chem.MolFromSmiles(smiles))

    # Pad with None for cases where not a square number to avoid IndexError
    mols = mols + [None] * (cols * rows - len(canonical_smiles))
    mols_grid = [[mols[i + j] for i in range(rows)] for j in range(0, cols * rows, rows)]

    if names:
        names = names + [""] * (cols * rows - len(canonical_smiles)) 
        names_grid = [[names[i + j] for i in range(rows)] for j in range(0, cols * rows, rows)]
    else: names_grid = None

    return Draw.MolsMatrixToGridImage(mols_grid, legendsMatrix=names_grid, **kwargs)

def visualise_3d_molecule_from_smiles(smiles_string: str):
    """
    Returns a 3D visualisation applet for a molecule from 
    a smiles string.
    """
    # Get mol from smile, explicitly add H;'s
    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)

    view = py3Dmol.view(
        data=Chem.MolToMolBlock(mol),  # Convert the RDKit molecule for py3Dmol
        style={"stick": {}, "sphere": {"scale": 0.3}}
        )
    view.zoomTo()

    return view

def save_model_weights(prefix, model, data):
    if not os.path.exists(os.path.join("../", "models")):
        os.makedirs(os.path.join("../", "models"))
        
    save_name = prefix + model.__class__.__name__ \
                + "-" + data.__class__.__name__ \
                + "-" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') \
                + ".pt"
    print(f"Saving model to {os.path.join('../', 'models', save_name)}")
    model.save_weights(os.path.join("../", "models", save_name))
    
    
# Model Sampling

def simple_generate(prefix: str, num_chars, model, indications_tensor, char_to_idx_mapping, idx_to_char_mapping, temperature = 0.0, device=None):
    """
    Simple character-by-character generation function.
    """

    def decode_indices_to_string(encoded_indices: list, idx_to_char_mapping: dict[int, str]):
        decoded = ''.join([idx_to_char_mapping[int(inx)] for inx in encoded_indices])
        return decoded

    def encode_string_to_indices(smiles_string: str, char_to_idx_mapping: dict[str, int]):
        encoded = [char_to_idx_mapping[c] for c in smiles_string]
        return encoded

    model.eval()
    generated = prefix
    
    with torch.no_grad():
        # Initialize state with indications
        state = model.init_state(indications_tensor.unsqueeze(0).to(device))  # Add batch dim

        # First, process the prefix to get the proper state
        if len(prefix) > 0:
            prefix_encoded = encode_string_to_indices(prefix, char_to_idx_mapping)
            prefix_tensor = torch.nn.functional.one_hot(
                torch.tensor(prefix_encoded), 
                num_classes=len(char_to_idx_mapping)
            ).float().to(device)
            
            # Process prefix through model to get proper state
            _, state = model(prefix_tensor.unsqueeze(0), state=state)
        
        # Now generate new characters one by one
        for i in range(num_chars - len(prefix)):
            # For generation, we need to feed the last character (or a dummy if this is the first step)
            if len(generated) > 0:
                last_char = generated[-1]
                last_char_idx = char_to_idx_mapping[last_char]
            else:
                # If no prefix, start with some default (this shouldn't happen with your use case)
                last_char_idx = 0
            
            # Create one-hot encoding for single character
            char_tensor = torch.nn.functional.one_hot(
                torch.tensor([last_char_idx]), 
                num_classes=len(char_to_idx_mapping)
            ).float().to(device)
            
            # Get prediction for next character
            output, state = model(char_tensor.unsqueeze(0), state=state)  # Add batch dim
            
            # Get most likely next token
            if temperature > 0:
                # Apply temperature scaling
                output = output / temperature
                probabilities = torch.softmax(output, dim=-1)
                next_token = torch.multinomial(probabilities[0, -1, :], num_samples=1).item()
            else:
                # Default to argmax if temperature is 0
                next_token = output[0, -1, :].argmax().item()
            
            # Decode and append
            next_char = decode_indices_to_string([next_token], idx_to_char_mapping)

            # TODO: Replace with vocab
            if next_char == '>' or next_char == '': # EOS token
            # if next_char == ' ' or next_char == '': # EOS token
                break

            generated += next_char
            
            # print(f"Step {i+1}: Added '{next_char}' -> '{generated}'")
    
    # TODO: TEMP: Replace with cvocab
    print("replacing")
    generated = generated.replace("<", "")


    return generated

def robust_generate(generate_function, max_attempts: int, **kwargs):
    n_chars = 100

    attempts = 0
    valid = False
    output = None

    while attempts < max_attempts and valid == False:
        output = generate_function(**kwargs)

        valid = validate_smiles_string(output)

        if valid:
            return output
        else:
            attempts += 1
        
    print(f"Could not generate valid molecular sample in {max_attempts} attemtps. Aborting.")
    return output