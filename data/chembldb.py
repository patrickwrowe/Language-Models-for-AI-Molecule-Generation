import pandas as pd
import pathlib
import attr
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation
from io import StringIO
from typing import Optional

# ToDo: Centralize
END_OF_MOLECULE_TOKEN = '[EOM]'

@attr.define
class ChemblDB:

    chemreps_filepath: pathlib.Path = pathlib.Path("../raw-data/chembldb/chembl_35_chemreps.txt.gz")
    tokenizer: Optional[Tokenizer] = None

    def _load_or_download(self, **kwargs):
        """TBD: Download file from source instead of needing to pre-download"""
        if not pathlib.Path.exists(self.chemreps_filepath):
            raise FileNotFoundError(f"${self.chemreps_filepath} was not found. Please download from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/")
        else:
            chembldb_chemreps_raw_pd = pd.read_table(
                pathlib.Path(self.chemreps_filepath), 
                compression="gzip",
                **kwargs,
            )
            return chembldb_chemreps_raw_pd

    def _preprocess(self, chemrepsdb: pd.DataFrame, column: str = "canonical_smiles"):
        db_column = chemrepsdb[column].to_list()
        text = END_OF_MOLECULE_TOKEN.join([col for col in db_column])
        return text

