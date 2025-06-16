from chembldb import ChemblDB
import pytest

@pytest.fixture(scope='session')
def chembldb_small():
    return ChemblDB()._load_or_download(nrows=25)

@pytest.fixture(scope='session')
def chembldb_small_preprocessed(chembldb_small):
    return ChemblDB()._preprocess(chembldb_small, column="canonical_smiles")

def test_chembldb_load(chembldb_small):
    assert len(chembldb_small) == 25

def test_chembldb_preprocess(chembldb_small_preprocessed, tmp_path):
    assert len(chembldb_small_preprocessed) > 1

def test_tokenize(chembldb_small_preprocessed, tmp_path):   
    tmp_txt_fn = "temp_chembldb_text.txt"

    with open(tmp_path / tmp_txt_fn, 'w') as f:
        f.write(chembldb_small_preprocessed)

    tokenizer = ChemblDB()._tokenize(str(tmp_path / tmp_txt_fn))
    assert tokenizer

def test_encode_molecule():
    pass