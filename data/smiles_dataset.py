from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import attr

@attr.define
class SMILESDataset(Dataset):
    """
    A PyTorch Dataset for tokenizing and encoding SMILES strings using a HuggingFace tokenizer.
    Attributes:
        smiles_list (list[str]): List of SMILES strings representing molecules.
        tokenizer (PreTrainedTokenizerFast): HuggingFace tokenizer for SMILES strings.
        max_length (int): Maximum length for tokenized SMILES sequences (default: 2048).
    Methods:
        __len__(): Returns the number of SMILES strings in the dataset.
        __getitem__(idx): Tokenizes and encodes the SMILES string at the given index, returning a dictionary
            containing input tensors for model training, including 'input_ids', 'attention_mask', and 'labels'.
    """

    # A list of smiles strings, e.g. ["CO", "CCC(C=O)"]
    smiles_list: list[str]

    # Hugginface tokenizer
    tokenizer: PreTrainedTokenizerFast

    # Max length of any individual SMILES id
    max_length: int = 2048

    # Batching and splitting
    batch_size: int = 64


    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        smiles = self.smiles_list[idx]

        # Seems super inefficient to encode each item one-by-one...
        # But I also don't want to store everything in memory.
        encoding = self.tokenizer.encode(
            smiles,
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
        )
        
        input_ids = torch.tensor(encoding, dtype=torch.long)

        one_hot = torch.nn.functional.one_hot(
            input_ids, 
            num_classes=len(self.tokenizer)
        )

        return one_hot
    
    def get_dataloader(self, train: bool):
        return DataLoader(
            self,
            batch_size = self.batch_size,
            shuffle = train
        )

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
