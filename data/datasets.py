from torch.utils.data import Dataset, DataLoader, Subset
import torch
import attr
import pandas as pd

from chembldb import ChemblDBIndications


@attr.define
class CharSMILESChEMBLIndications(Dataset):
    """
    Character level SMILES database with disease indications
    """

    all_data: pd.DataFrame = ChemblDBIndications()._preprocess(ChemblDBIndications()._load_data())
    
    # Length and batch size for loading
    max_length: int = 64
    batch_size: int = 128
    frac_train: float = 0.8

    char_to_idx: dict[str, int] = attr.field(init=False)
    idx_to_char: dict[int, str] = attr.field(init=False)

    def __attrs_post_init__(self):
        self.all_smiles: list[str] = self.all_data["canonical_smiles"].tolist()
        self.characters: list[str] = list(set(''.join(self.all_smiles)))

        self.char_to_idx = {c: inx for inx, c in enumerate(self.characters)}
        self.idx_to_char = {inx: c for inx, c in enumerate(self.characters)}

        self.encoded_smiles = [[self.char_to_idx[c] for c in smiles_string] for smiles_string in self.all_smiles]

        get_tensor = lambda x: torch.tensor(x.values.astype(float), dtype=torch.float32)
        self.indications_tensor = get_tensor(self.all_data.drop(columns=["canonical_smiles"]))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx: int):

        indications = self.indications_tensor[idx, :]

        encoded_smiles = torch.tensor(self.encoded_smiles[idx])

        smiles_one_hot = torch.nn.functional.one_hot(
            encoded_smiles, 
            num_classes=len(self.characters)
            ).float()
        
        return smiles_one_hot, indications, encoded_smiles

    def get_dataloader(self, train: bool):
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

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
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

    length: int = 32

    # Batching and splitting
    batch_size: int = 128

    all_smiles: str = attr.field(init=False)
    encoded_smiles: torch.Tensor = attr.field(init=False)
        
    def __attrs_post_init__(self):
        self.all_smiles = '<EOM>'.join(self.smiles_list)

        # Efficient but causes memory issues for large datasets
        # Requires clever caching to work properly
        self.encoded_smiles = self.tokenizer.encode(self.all_smiles)

    def __len__(self):
        return len(self.encoded_smiles) // self.length

    def __getitem__(self, idx: int):
        start = idx * self.length 
        end = start + self.length
        encoding = self.encoded_smiles[start:end + 1]
        text_tensor = self.encoding_to_one_hot(encoding[:-1])
        
        target_indices = torch.tensor(encoding[1:], dtype=torch.long)
        return text_tensor, target_indices

    def encode_smiles_to_one_hot(self, smiles) -> torch.Tensor:
        return self.encoding_to_one_hot(self.encode_smiles(smiles))

    def encode_smiles(self, smiles):
        # Seems super inefficient to encode each item one-by-one...
        # But I also don't want to store everything in memory.
        encoding = self.tokenizer.encode(
            smiles,
            truncation=True,
            max_length=self.length,
            padding='max_length',
        )
        
        return encoding
    
    def encoding_to_one_hot(self, encoding):
        input_ids = torch.tensor(encoding, dtype=torch.long)

        one_hot = torch.nn.functional.one_hot(
            input_ids, 
            num_classes=len(self.tokenizer)
        ).type(torch.float32)

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

    def __len__(self):
        return len(self.all_smiles) // self.length

    def __getitem__(self, idx: int):
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

    def get_dataloader(self, train: bool):
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

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    