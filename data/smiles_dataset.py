from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from torch.utils.data import Dataset
import torch
import attr

@attr.define
class SMILESDataset(Dataset):

    # A list of smiles strings, e.g. ["CO", "CCC(C=O)"]
    smiles_list: list[str]

    # Hugginface tokenizer
    tokenizer: PreTrainedTokenizerFast

    # Max length of any individual SMILES id
    max_length: int = 2048

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        encoding = self.tokenizer(
            smiles,
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
            return_tensors = 'pt'
        )
        
        if not encoding or "input_ids" not in encoding:
            raise ValueError(f"Failed to encode: {smiles}")

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = item["input_ids"].clone()

        return item



