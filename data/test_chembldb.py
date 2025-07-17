from chembldb import ChemblDB
from chembed_tokenize import train_bpe_tokenizer
import pytest
from tokenizers.models import BPE
from tokenizers import Tokenizer

TEST_CHEMBLDB_NROWS = 25

@pytest.fixture(scope='session')
def chembldb_small():
    return ChemblDB()._load_or_download(nrows=TEST_CHEMBLDB_NROWS)

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
    assert len(chembldb_small) == TEST_CHEMBLDB_NROWS

def test_chembldb_preprocess(chembldb_small_preprocessed):

    txt, filename = chembldb_small_preprocessed

    # When split on EOM token, should get nrows back.
    assert len(txt.split("[EOM]")) == TEST_CHEMBLDB_NROWS

def test_train_tokenizer(chembldb_small_preprocessed, tmp_path):   

    _, filename = chembldb_small_preprocessed

    tokenizer = train_bpe_tokenizer([str(filename)])
    
    # Check more things! 

    assert tokenizer

def test_encode_molecules(chembl_bpe_tokenizer, smiles_strings):
    encoded = [chembl_bpe_tokenizer.encode(smiles) for smiles in smiles_strings]

def test_load_tokenizer(smiles_strings):
    tokenizer = Tokenizer.from_file("./tokenizers/tokenizer-chembldb-16-06-2025.json")
    encoded = tokenizer.encode(smiles_strings[0])
    assert encoded