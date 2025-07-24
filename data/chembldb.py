import pandas as pd
import pathlib
import attr
import sqlite3


CHEMBL_DB_PATH = "../raw-data/chembldb/chembl_35/chembl_35_sqlite/chembl_35.db"

SQL_DRUG_INDICATION_QUERY = """
    SELECT comp.molregno, comp.canonical_smiles, mol.indication_class, rec.record_id, ind.mesh_heading, ind.max_phase_for_ind
    FROM compound_structures as comp
    INNER JOIN molecule_dictionary as mol
    ON comp.molregno = mol.molregno
    INNER JOIN compound_records as rec
    on mol.molregno = rec.molregno
    INNER JOIN drug_indication as ind
    ON rec.record_id = ind.record_id
"""

# ToDo: Centralize
END_OF_MOLECULE_TOKEN = '[EOM]'

@attr.define
class ChemblDBChemreps:
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
            raise FileNotFoundError(f"{self.chemreps_filepath} was not found. Please download from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/")
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


@attr.define 
class ChemblDBIndications:
    """
    A class for extracting the drug indications, alongside relevant smiles
    strings from the chembldb database.
    """

    query_df: pd.DataFrame = attr.field(init=False)

    query: str = SQL_DRUG_INDICATION_QUERY

    def _load_data(self):
        if not pathlib.Path.exists(pathlib.Path(CHEMBL_DB_PATH)):
            raise FileNotFoundError(f"{CHEMBL_DB_PATH} was not found")
        else:
            con = sqlite3.connect(CHEMBL_DB_PATH)
            cur = con.cursor()
            pd_df = pd.read_sql_query(sql=SQL_DRUG_INDICATION_QUERY, con=con)
            con.close()
            return pd_df

    def _preprocess(self, raw_df: pd.DataFrame):
        raw_df = raw_df[raw_df.max_phase_for_ind >= 3].drop(  # Don't want to train on things which weren't efficacious
                            columns=[
                            'molregno', 
                            'record_id', 
                            'max_phase_for_ind', 
                            'indication_class'
                        ]  # Don't need identifiers etc
                    )
        
        pd.get_dummies(raw_df, columns=['mesh_heading']) # One-hot like for disease indications

        return raw_df

        