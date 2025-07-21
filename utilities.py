
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

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


def generate_sequence(prefix, num_chars, model, dataset, device=None, temperature=1.0):
    """
    Generate a sequence of characters using the RNN model.
    
    Args:
        prefix (str): Initial string to start generation from
        num_chars (int): Number of characters to generate
        model: The trained RNN model
        dataset: Dataset containing tokenizer and encoding methods
        device: Device to run inference on
        temperature (float): Sampling temperature (1.0 = deterministic, >1.0 = more random)
    
    Returns:
        str: Generated sequence including the prefix
    """
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        generated = prefix
        
        for _ in range(num_chars):
            # Encode the current generated text
            current_encoded = dataset.encoding_to_one_hot(dataset.tokenizer.encode(generated))
            current_tensor = torch.tensor(current_encoded, device=device, dtype=torch.float32)
            
            # Get model prediction
            output = model(current_tensor.unsqueeze(0))  # Add batch dimension
            
            # Get the last timestep's logits
            last_logits = output[0, -1, :]  # [vocab_size]
            
            # temperature probability
            last_logits = last_logits / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Decode the token and add to generated text
            next_char = dataset.tokenizer.decode([next_token_id])
            generated += next_char
            
            # Limit length to prevent runaway generation
            if len(generated) > len(prefix) + num_chars * 2:
                break
                
    return generated

def simple_generate(prefix, num_chars, model, dataset, device=None):
    """
    Simple character-by-character generation function.
    """
    model.eval()
    generated = prefix
    
    with torch.no_grad():
        for i in range(num_chars):
            # Encode current text
            encoded = dataset.encoding_to_one_hot(dataset.tokenizer.encode(generated))
            input_tensor = torch.tensor(encoded, device=device, dtype=torch.float32)
            
            # Get prediction
            output = model(input_tensor.unsqueeze(0))  # Add batch dim
            
            # Get most likely next token
            next_token = output[0, -1, :].argmax().item()
            
            # Decode and append
            next_char = dataset.tokenizer.decode([next_token])
            generated += next_char
            
            print(f"Step {i+1}: Added '{next_char}' -> '{generated}'")
            
    return generated

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

    