from torch.utils.data import Dataset, DataLoader, Subset
import torch
import attr
import pandas as pd

import sys
sys.path.append("..")

from data.chembldb import ChemblDBIndications
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

@attr.define
class CharSMILESChEMBLIndications(Dataset):
    """
    Character level SMILES database with disease indications
    """

    # Length and batch size for loading
    max_length: int = 512

    all_data: pd.DataFrame = ChemblDBIndications()._preprocess(ChemblDBIndications()._load_data(), max_length=max_length)
    
    batch_size: int = 64
    frac_train: float = 0.8

    char_to_idx: dict[str, int] = attr.field(init=False)
    idx_to_char: dict[int, str] = attr.field(init=False)

    # account for padding, to be more cleanly handled with tokenizers
    padding_index = 0
    padding_char = " "  
    end_char = "£"

    smiles_column_title = "canonical_smiles"

    def __attrs_post_init__(self):
        self.all_smiles: list[str] = self.all_data[self.smiles_column_title].tolist()

        self.all_smiles = [smiles + self.end_char for smiles in self.all_smiles]  # Add end character to each SMILES string or molecules never end
        self.characters: list[str] = list(set(''.join(self.all_smiles)))
        self.characters.extend([
            self.padding_char
        ])

        # This will only make sense if padding index is 0
        # This class is soon to be superceded with a variant using proper tokenizers 
        # So I won't worry about writing better logic to handle different padding styles
        assert self.padding_index == 0

        # Create char to index mappings, RESERVE index 0 for padding
        self.char_to_idx = {c: inx + 1 for inx, c in enumerate(self.characters)}
        self.idx_to_char = {inx + 1: c for inx, c in enumerate(self.characters)}
        
        # padding characters
        self.char_to_idx[self.padding_char] = self.padding_index
        self.idx_to_char[self.padding_index] = self.padding_char

        # Encode all smiels strings and pad to appropriate length
        self.encoded_smiles = [self.encode_smiles_string(smiles_string) for smiles_string in self.all_smiles]

        # ToDo: Clean up this mess
        get_tensor = lambda x: torch.tensor(x.values.astype(float), dtype=torch.float32)
        self.indications_names = self.all_data.columns.drop(self.smiles_column_title).to_list()
        self.indications_tensor = get_tensor(self.all_data.drop(columns=[self.smiles_column_title]))
        
        # Shortcuts for sizes
        self.vocab_size = len(self.char_to_idx)
        self.num_indications = self.indications_tensor.shape[-1]

    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        indications = self.indications_tensor[idx, :]

        input_seq = self.encoded_smiles[idx][:-1]
        target_seq = self.encoded_smiles[idx][1:]

        smiles_one_hot = torch.nn.functional.one_hot(
            input_seq, 
            num_classes=len(self.characters)
        ).float()
        
        return smiles_one_hot, indications, target_seq
    
    def encode_smiles_string(self, smiles: str) -> torch.Tensor:
        """
        Encodes a smiles string to a list of integers, pads to self.max_length with pad character.
        """
        encoded_smiles = [self.char_to_idx[c] for c in smiles]

        # Not needed as we now pad durign collation
        # if len(encoded_smiles) <= self.max_length:
        #     encoded_smiles.extend([self.padding_index] * (self.max_length - len(encoded_smiles)))
        # elif len(encoded_smiles) > self.max_length:
        #     # raise ValueError("SMILES String longer than defined max length.")
        #     encoded_smiles = encoded_smiles[:512]

        return torch.tensor(encoded_smiles)
    
    def get_indications_tensor(self, indication: str):
        indication_index = self.indications_names.index(indication)
        indication_tensor = torch.zeros(len(self.indications_names))
        indication_tensor[indication_index] = 1
        return indication_tensor

    def get_dataloader(self, train: bool) -> DataLoader:
        # Create proper train/val split
        total_len = len(self)
        train_len = int(self.frac_train * total_len)  # 80% for training
        
        if train:
            subset = Subset(self, range(train_len))
        else:
            subset = Subset(self, range(train_len, total_len))
            
        return DataLoader(
            subset,
            batch_size = self.batch_size,
            collate_fn=self.collate_fn,
            shuffle = train
        )

    def collate_fn(self, batch):
        """
        Custom collate function to handle variable length sequences and pad them.
        """
        smiles, indications, targets = zip(*batch)

        # Pad the SMILES sequences
        smiles_padded = pad_sequence(smiles, batch_first=True, padding_value=self.padding_index)
        
        # Pad the target sequences
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=self.padding_index)

        # Stack the indications
        indications_stacked = torch.stack(indications)

        return smiles_padded, indications_stacked, targets_padded

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(train=False)

    

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import attr

@attr.define
class SMILESDatasetContinuous(Dataset):
    """
    A PyTorch Dataset for tokenizing and encoding SMILES strings using a HuggingFace tokenizer.
    Attributes:
        smiles_list (list[str]): List of SMILES strings representing molecules.
        tokenizer (PreTrainedTokenizerFast): HuggingFace tokenizer for SMILES strings.
        max_length (int): Maximum length for tokenized SMILES sequences.
    Methods:
        __len__(): Returns the number of SMILES strings in the dataset.
        __getitem__(idx): Tokenizes and encodes the SMILES string at the given index
    """

    # A list of smiles strings, e.g. ["CO", "CCC(C=O)"]
    smiles_list: list[str]
    
    # Hugginface tokenizer
    tokenizer: PreTrainedTokenizerFast

    length: int = 32

    # Batching and splitting
    batch_size: int = 128
    frac_train: float = 0.8

    all_smiles: str = attr.field(init=False)
    encoded_smiles: torch.Tensor = attr.field(init=False)
        
    def __attrs_post_init__(self):
        self.all_smiles = '<EOM>'.join(self.smiles_list)

        # Efficient but causes memory issues for large datasets
        # Requires clever caching to work properly
        self.encoded_smiles = torch.tensor(self.tokenizer.encode(self.all_smiles))

    def __len__(self) -> int:
        return len(self.encoded_smiles) // self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.length 
        end = start + self.length
        encoding = self.encoded_smiles[start:end + 1]

        input_seq = encoding[:-1]
        target_seq = encoding[1:]

        text_tensor = self.encoding_to_one_hot(input_seq)
        target_indices = torch.tensor(target_seq, dtype=torch.long)

        return text_tensor, target_indices

    def encode_smiles_to_one_hot(self, smiles) -> torch.Tensor:
        return self.encoding_to_one_hot(self.encode_smiles(smiles))

    def encode_smiles(self, smiles) -> list[int]:
        # Seems super inefficient to encode each item one-by-one...
        # But I also don't want to store everything in memory.
        encoding = self.tokenizer.encode(
            smiles,
            truncation=True,
            max_length=self.length,
            padding='max_length',
        )
        
        return encoding
    
    def encoding_to_one_hot(self, encoding) -> torch.Tensor:
        input_ids = torch.tensor(encoding, dtype=torch.long)

        one_hot = torch.nn.functional.one_hot(
            input_ids, 
            num_classes=len(self.tokenizer)
        ).type(torch.float32)

        return one_hot
     
    def get_dataloader(self, train: bool) -> DataLoader:
        # Create proper train/val split
        total_len = len(self)
        train_len = int(self.frac_train * total_len)  # 80% for training
        
        if train:
            subset = Subset(self, range(train_len))
        else:
            subset = Subset(self, range(train_len, total_len))
            
        return DataLoader(
            subset,
            batch_size = self.batch_size,
            shuffle = train
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(train=False)


@attr.define
class CharacterLevelSMILES(Dataset):
    """
    A PyTorch Dataset for character-level tokenization of SMILES strings.
    Attributes:
        smiles_list (list[str]): List of SMILES strings representing molecules.
        tokenizer (PreTrainedTokenizerFast): HuggingFace tokenizer for SMILES strings.
        max_length (int): Maximum length for tokenized SMILES sequences (default: 2048).
    """

    smiles_list: list[str]

    # Length and batch size for loading
    length: int = 64
    batch_size: int = 128
    frac_train: float = 0.8

    # All smiles strings concatenated into one string
    all_smiles: str = attr.field(init=False)
    # As above but encoded as indices
    encoded_smiles: list[int] = attr.field(init=False)
    
    char_to_idx: dict[str, int] = attr.field(init=False)
    idx_to_char: dict[int, str] = attr.field(init=False)

    def __attrs_post_init__(self):
        self.all_smiles: str = ' '.join(self.smiles_list)
        self.characters: list[str] = list(set(self.all_smiles))

        self.char_to_idx = {c: inx for inx, c in enumerate(self.characters)}
        self.idx_to_char = {inx: c for inx, c in enumerate(self.characters)}

        self.encoded_smiles = [self.char_to_idx[c] for c in self.all_smiles]

        # To account for single character shifted input/target indexing 
        self.length += 1

    def __len__(self) -> int:
        return len(self.all_smiles) // self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.length
        end = start + self.length 
        
        # Get the character indices for this slice
        if end >= len(self.encoded_smiles):
            end = len(self.encoded_smiles)
        
        sequence = self.encoded_smiles[start:end]
        
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
    
        # Convert to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        
        # Convert input to one-hot encoding, but leave the targets as is...
        input_one_hot = torch.nn.functional.one_hot(input_tensor, num_classes=len(self.characters)).float()
        
        return input_one_hot, target_tensor

    def get_dataloader(self, train: bool) -> DataLoader:
        # Create proper train/val split
        total_len = len(self)
        train_len = int(self.frac_train * total_len)  # 80% for training
        
        if train:
            subset = Subset(self, range(train_len))
        else:
            subset = Subset(self, range(train_len, total_len))
            
        return DataLoader(
            subset,
            batch_size = self.batch_size,
            shuffle = train
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(train=False)

    def decode_indices_to_string(self, encoded_indices: list):
        decoded = ''.join([self.idx_to_char[int(inx)] for inx in encoded_indices])
        return decoded

    def encode_string_to_indices(self, smiles_string: str):
        encoded = [self.char_to_idx[c] for c in smiles_string]
        return encoded

    