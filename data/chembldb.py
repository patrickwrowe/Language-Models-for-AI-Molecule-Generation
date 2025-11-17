import pandas as pd
import pathlib
import attrs
import sqlite3
from typing import Optional

CHEMBL_DB_PATH = "../../data/chembl/chembl_35/chembl_35_sqlite/chembl_35.db"

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

SQL_ALL_MOL_QUERY = """
    SELECT canonical_smiles FROM compound_structures
"""

def set_chembldb_path(path: str):
    CHEMBL_DB_PATH = path

@attrs.define
class ChemblDBData:
    query_df: Optional[pd.DataFrame] = attrs.field(init=False)

    query: str = attrs.field()

    def _load_data(self):
        if not pathlib.Path.exists(pathlib.Path(CHEMBL_DB_PATH)):
            raise FileNotFoundError(f"{CHEMBL_DB_PATH} was not found")
        else:
            con = sqlite3.connect(CHEMBL_DB_PATH)
            pd_df = pd.read_sql_query(sql=self.query, con=con)
            con.close()
            return pd_df

    def _preprocess(self, *args, **kwargs):
        raise NotImplementedError()

    def load(self, *args, **kwargs):
        raise NotImplementedError()

@attrs.define
class ChemblDBChemreps(ChemblDBData):
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

    query: str = SQL_ALL_MOL_QUERY

    def _preprocess(self, chemrepsdb: pd.DataFrame, column: str = "canonical_smiles", max_length: int = 256, save_path: Optional[str] = None):
        
        preprocessed = chemrepsdb[chemrepsdb[column].str.len().lt(max_length)]
        preprocessed = preprocessed.drop_duplicates()

        if save_path:
            # Save the file
            preprocessed.to_csv(save_path, index=False)
        
        return preprocessed

    @classmethod
    def load(cls, preprocessed_filename: Optional[str] = None, save_if_missing: bool = True):
        if preprocessed_filename and pathlib.Path.exists(pathlib.Path(preprocessed_filename)):
            print(f"loading {cls.__name__} dataset from file")
            return pd.read_csv(preprocessed_filename)
        else:
            print("Preprocessed data not found... attemping to load and preprocess")
            return cls()._preprocess(cls()._load_data(), save_path=preprocessed_filename if save_if_missing else None)


@attrs.define 
class ChemblDBIndications(ChemblDBData):
    """
    A class for extracting the drug indications, alongside relevant smiles
    strings from the chembldb database.
    """

    query: str = SQL_DRUG_INDICATION_QUERY

    def _preprocess(self, raw_df: pd.DataFrame, max_length: int = 1024, min_phase_for_ind: int = 3, column: str = "canonical_smiles", save_path: Optional[str] = None):
        raw_df = raw_df[raw_df.max_phase_for_ind >= min_phase_for_ind].drop(  # Don't want to train on things which weren't efficacious
                            columns=[
                            'molregno', 
                            'record_id', 
                            'max_phase_for_ind', 
                            'indication_class'
                        ]  # Don't need identifiers etc
                    )
        
        # Drop smiles with string lenth longer than max_length
        raw_df = raw_df[raw_df[column].str.split().str.len().lt(max_length)]

        preprocessed_df = pd.get_dummies(raw_df, columns=['mesh_heading']) # One-hot like for disease indications

        # Drop rarely occuring indications, replace with Other
        min_indications = 50
        indication_counts = preprocessed_df.filter(like='mesh_heading_').sum()
        rare_indications = indication_counts[indication_counts < min_indications].index.tolist()
        if rare_indications:
            preprocessed_df['mesh_heading_Other'] = preprocessed_df[rare_indications].sum(axis=1).astype(bool)
            preprocessed_df.drop(columns=rare_indications, inplace=True)

        if save_path:
            # Save the file
            preprocessed_df.to_csv(save_path, index=False)

        return preprocessed_df

    @classmethod
    def load(cls, preprocessed_filename: Optional[str] = None, save_if_missing: bool = True):
        if preprocessed_filename and pathlib.Path.exists(pathlib.Path(preprocessed_filename)):
            print(f"loading {cls.__name__} dataset from file")
            return pd.read_csv(preprocessed_filename)
        else:
            print("Preprocessed data not found... attemping to load and preprocess")
            return cls()._preprocess(cls()._load_data(), save_path=preprocessed_filename if save_if_missing else None)