import pandas as pd
import pathlib

def load_chembldb(filepath: str):

    fp = pathlib.Path(filepath)
    chembldb_raw_pd = pd.read_csv(fp, sep=" ")
