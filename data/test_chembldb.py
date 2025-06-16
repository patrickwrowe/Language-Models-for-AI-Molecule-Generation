from chembldb import ChemblDB
from chembed_tokenize import train_bpe_tokenizer
import pytest

@pytest.fixture(scope='session')
def chembldb_small():
    return ChemblDB()._load_or_download(nrows=25)

@pytest.fixture(scope='session')
def chembldb_small_preprocessed(tmp_path_factory, chembldb_small):
    tmp_txt_fn = "temp_chembldb_text.txt"

    preprocessed_text = ChemblDB()._preprocess(chembldb_small, column="canonical_smiles")
    tmp_path = tmp_path_factory.mktemp("txt")

    filename = tmp_path / tmp_txt_fn
    with open(filename, 'w') as f:
        f.write(preprocessed_text)

    return preprocessed_text, filename

@pytest.fixture(scope="session")
def chembl_bpe_tokenizer(chembldb_small_preprocessed):
    
    _, filename = chembldb_small_preprocessed

    tokenizer = train_bpe_tokenizer([str(filename)])
    return tokenizer

@pytest.fixture()
def smiles_strings():
    return [
        "COc1c(O)cc(O)c(C(=N)Cc2ccc(O)cc2)c1O"
    ]

def test_chembldb_load(chembldb_small):
    assert len(chembldb_small) == 25

def test_chembldb_preprocess(chembldb_small_preprocessed):

    txt, filename = chembldb_small_preprocessed

    assert len(chembldb_small_preprocessed) > 1

def test_train_tokenizer(chembldb_small_preprocessed, tmp_path):   

    _, filename = chembldb_small_preprocessed

    tokenizer = train_bpe_tokenizer([str(filename)])
    
    # Check more things! 

    assert tokenizer

def test_encode_molecule(chembl_bpe_tokenizer, smiles_strings):
    encoded = chembl_bpe_tokenizer.encode(smiles_strings[0])
    print(f"Encoded Molecule: {encoded}")