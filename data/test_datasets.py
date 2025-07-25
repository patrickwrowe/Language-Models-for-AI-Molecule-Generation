import pytest

from datasets import CharacterLevelSMILES, SMILESDatasetContinuous, CharSMILESChEMBLIndications
from chembed_tokenize import load_chembed_tokenizer
from torch.utils.data import DataLoader
import torch
import pandas as pd

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

@pytest.fixture(scope="session")
def dummy_indications(smiles_strings):
    indications_headers = ["Hungry", "Bubonic Plague", "Flu"]

    # Get a fake one-hot encoded array of indications
    indications_tensor = torch.zeros((len(smiles_strings), len(indications_headers)))
    indices = torch.randint(low=0, high=len(indications_headers), size=(len(smiles_strings),))
    for i, sample in enumerate(indications_tensor):
        indications_tensor[i][indices[i]] = 1

    data = dict({"canonical_smiles": smiles_strings}, **{indication: vector for indication, vector in zip(indications_headers, indications_tensor.T)})

    df = pd.DataFrame(
       data = data
    )

    return df

def check_onehot(tensor: torch.Tensor):
    # check elements are one-hot
    if not torch.all((tensor == 0) | (tensor == 1)):
        return False
    
    return True

def test_create_smilesdatasetcontinuous_dataloader(smiles_strings):

    assert smiles_strings and all(isinstance(s, str) for s in smiles_strings)

    tokenizer = load_chembed_tokenizer()
    dataset = SMILESDatasetContinuous(
        smiles_strings, 
        tokenizer
    )

    # Get one, one-hot encoded string
    one_encoded = dataset[0][0]

    # should return a pytorch tensor
    assert type(one_encoded) == torch.Tensor, f"Expected {torch.Tensor} got {type(one_encoded)}"

    # Check one-hot has len(seq) x len(tokens) dims
    assert one_encoded.dim() == 2 

    assert check_onehot(one_encoded)

    # Check that we can load it into a dataloader
    test_batch_size = 8
    dataloader = dataset.train_dataloader()

    batch = next(iter(dataloader))

    assert len(batch) == 2

    assert list(batch[0].shape) == [test_batch_size, dataset.length, len(tokenizer)]
    assert list(batch[1].shape) == [test_batch_size, dataset.length]


def test_create_characterlevelsmiles_dataloader(smiles_strings):

    assert smiles_strings and all(isinstance(s, str) for s in smiles_strings)

    test_length = 8
    test_batch_size = 4
    dataset = CharacterLevelSMILES(smiles_strings, length = test_length, batch_size = test_batch_size)

    # test __getitem__ method
    one_encoded = dataset[0]

    # Check dimensions
    assert one_encoded[0].dim() == 2
    assert one_encoded[1].dim() == 1

    # first dim is sequence length
    assert one_encoded[0].shape[0] == test_length
    assert one_encoded[1].shape[0] == test_length
    
    # one hot dimension
    assert one_encoded[0].shape[1] == len(dataset.char_to_idx)

    dataloader = dataset.train_dataloader()
    one_batch = next(iter(dataloader))

    assert len(one_batch) == 2
    assert list(one_batch[0].shape) == [test_batch_size, test_length, len(dataset.char_to_idx)]
    assert list(one_batch[1].shape) == [test_batch_size, test_length]
    
def test_create_charsmilescchemblindications(dummy_indications):
    
    test_batch_size = 4
    dataset = CharSMILESChEMBLIndications(all_data = dummy_indications, batch_size=test_batch_size)
    one_encoded = dataset[0]
    assert len(one_encoded) == 3
    assert one_encoded[0].shape[-1] == len(dataset.char_to_idx)
    # Indications shoudl be len columns - SMILES column
    assert one_encoded[1].shape[0] == len(dummy_indications.columns) - 1

    dataloader = dataset.train_dataloader()
    one_batch = next(iter(dataloader))

    assert len(one_batch) == 3
    assert one_encoded[0].shape[-1] == len(dataset.char_to_idx)
    assert one_encoded[0].shape[0] == test_batch_size

    breakpoint()
    