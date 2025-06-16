import pandas as pd
import pathlib
import attr
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation
from io import StringIO

@attr.define
class ChemblDB:

    chemreps_filepath: pathlib.Path = pathlib.Path("../raw-data/chembldb/chembl_35_chemreps.txt.gz")

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

    def _preprocess(self, chemrepsdb, column: str = "canonical_smiles"):
        db_column = chemrepsdb[column].to_list()
        text = '[EOM]'.join([col for col in db_column])
        return text

    def _tokenize(self, filepath, vocab_size=1024):
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        trainer = BpeTrainer(special_tokens=['[UNK]', '[EOM]'], vocab_size=vocab_size)

        # Do we need more special tokens?
        # special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

        # Include pre-tokenizer for punctuation
        tokenizer.pre_tokenizer = Punctuation()

        tokenizer.train(
            [filepath],
            trainer,
        )

        return tokenizer