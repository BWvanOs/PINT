from __future__ import annotations

import pandas as pd
from pint_app.neighborhood.params import coerce_and_validate_celldata_df

def load_and_validate_csv(path: str) -> pd.DataFrame:
    """
    Read CSV from path input
    
    output trimmed DF and raises error is column is missing
    """
    df = pd.read_csv(path)
    return coerce_and_validate_celldata_df(df)
