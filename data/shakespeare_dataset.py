import attr
from torch.utils.data import Dataset, DataLoader
import torch

@attr.define
class characterLevelShakespeare(Dataset):

    # All smiles strings concatenated into one string
    all_text: str 

    # Length and batch size for loading
    length: int = 1024
    batch_size: int = 128

    # As above but encoded as indices
    encoded_text: list[int] = attr.field(init=False)
    
    char_to_idx: dict[str, int] = attr.field(init=False)
    idx_to_char: dict[int, str] = attr.field(init=False)

    def __attrs_post_init__(self):
        
        self.characters: list[str] = list(set(self.all_text))

        self.char_to_idx = {c: inx for inx, c in enumerate(self.characters)}
        self.idx_to_char = {inx: c for inx, c in enumerate(self.characters)}

        self.encoded_text = [self.char_to_idx[c] for c in self.all_text]

    def __len__(self):
        return len(self.all_text) // self.length

    def __getitem__(self, idx: int):
        start = idx * self.length
        end = start + self.length
        
        # Get the character indices for this slice
        if end >= len(self.encoded_text):
            end = len(self.encoded_text)
        
        sequence = self.encoded_text[start:end]
        
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
    
        # Convert to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        
        # Convert input to one-hot encoding, but leave the targets as is...
        input_one_hot = torch.nn.functional.one_hot(input_tensor, num_classes=len(self.characters)).float()
        
        return input_one_hot, target_tensor

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

    