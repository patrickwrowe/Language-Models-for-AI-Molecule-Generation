from chembldb import ChemblDB
import pytest

@pytest.fixture(scope='session')
def chembldb_small():
    return ChemblDB()._load_or_download(nrows=25)

def test_chembldb_load(chembldb_small):
    assert len(chembldb_small) == 25

@pytest.fixture(scope='session')
def test_chembldb_preprocess(chembldb_small):
    text_preprocessed = ChemblDB()._preprocess(chembldb_small, column="canonical_smiles")
    assert len(text_preprocessed) > 1
    return text_preprocessed

def test_tokenize(test_chembldb_preprocess):
    tokens = ChemblDB()._tokenize(test_chembldb_preprocess)
    assert tokens

