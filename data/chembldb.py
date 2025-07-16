import pandas as pd
import pathlib
import attr

# ToDo: Centralize
END_OF_MOLECULE_TOKEN = '[EOM]'

@attr.define
class ChemblDB:
    """
    A class for handling ChEMBL database chemical representations.

    Attributes:
        chemreps_filepath (pathlib.Path): Path to the ChEMBL chemical representations file.

    Methods:
        _load_or_download(**kwargs):
            Loads the chemical representations file into a pandas DataFrame.
            Raises FileNotFoundError if the file does not exist.
            Note: Download functionality is to be implemented.

        _preprocess(chemrepsdb: pd.DataFrame, column: str = "canonical_smiles"):
            Preprocesses the chemical representations DataFrame by extracting the specified column,
            joining its entries with END_OF_MOLECULE_TOKEN, and returning the resulting text.
    """

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

    def _preprocess(self, chemrepsdb: pd.DataFrame, column: str = "canonical_smiles"):
        db_column = chemrepsdb[column].to_list()
        text = END_OF_MOLECULE_TOKEN.join([col for col in db_column])
        return text

