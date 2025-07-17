import pytest

from smiles_dataset import SMILESDataset
from chembed_tokenize import load_chembed_tokenizer
from torch.utils.data import DataLoader
import torch

@pytest.fixture(scope="session")
def smiles_strings():
    smiles_strings = [
        "CC",
        "C=C",
        "CCCCCC",
        "CCO",
        "c1ccccc1",
        "c1cc(CCC)ccc1",
        "CN=C=O",
        "O=Cc1ccc(O)c(OC)c1COc1cc(C=O)ccc1O",
        "CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1",
        "CC(=O)NCCC1=CNc2c1cc(OC)cc2CC(=O)NCCc1c[nH]c2ccc(OC)cc12",
        "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5",
        "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N",
        "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)1",
        "CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2",
        "CC(=O)OCCC(/C)=C\\C[C@H](C(C)=C)CCC=C",
        "CCC[C@@H](O)CC\\C=C\\C=C\\C#CC#C\\C=C\\COCCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO"
    ]

    print("Testing with SMILES:", len(smiles_strings))
    for i, s in enumerate(smiles_strings):
        print(f"{i}: {s}")

    return smiles_strings

def test_create_dataloader(smiles_strings):

    assert smiles_strings and all(isinstance(s, str) for s in smiles_strings)

    tokenizer = load_chembed_tokenizer()
    dataset = SMILESDataset(
        smiles_strings, 
        tokenizer
    )

    one_encoded = dataset[0]

    # should return a pytorch tensor
    assert type(one_encoded) == torch.Tensor, f"Expected {torch.Tensor} got {type(one_encoded)}"

    # Check one-hot has len(seq) x len(tokens) dims
    assert one_encoded.dim() == 2 

    # check elements are one-hot
    assert torch.all((one_encoded == 0) | (one_encoded == 1))

    # Check that we can load it into a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size = 8,
        shuffle=False
    )

    batch = next(iter(dataloader))

    assert batch.dim() == 3

    assert list(batch.shape) == [8, dataset.max_length, len(tokenizer)]