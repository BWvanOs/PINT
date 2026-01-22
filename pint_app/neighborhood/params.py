from __future__ import annotations
import pandas as pd

##Centralize what exactly are required column as this might change in the future.
REQUIRED_COLUMNS = [
    "CellID",
    "X_Location",
    "Y_Location",
    "ClusterName",
    "SampleNumber",
    "Condition",
]

CELLDATA_DTYPES: dict[str, str] = {
    "CellID": "string",
    "X_Location": "float64",
    "Y_Location": "float64",
    "ClusterName": "string",
    "SampleNumber": "string",
    "Condition": "string",
}

EXAMPLE_VALUES: dict[str, list] = {
    "CellID": [1, 2, 3, 4, 5, 6],
    "X_Location": [120.5, 150.2, 180.1, 45.0, 60.3, 90.7],
    "Y_Location": [80.2, 95.1, 110.4, 210.0, 195.6, 205.2],
    "ClusterName": ["T cell", "T cell", "Myeloid", "Stroma", "Stroma", "T cell"],
    "SampleNumber": [1, 1, 1, 2, 2, 2],
    "Condition": ["Ctrl", "Ctrl", "Ctrl", "Treat", "Treat", "Treat"],
}

NUMERIC_COLUMNS = ["X_Location", "Y_Location"]
STRING_COLUMNS = ["CellID", "ClusterName", "SampleNumber", "Condition"]

#make it clear to the user which dataframe it needs to input. The sample data is pulled from this function so adding new column will remain consistent throughout
def example_input_df(n: int = 6) -> pd.DataFrame: ##Make empty DF with 6 rows
    data = {}
    for col in REQUIRED_COLUMNS:
        if col in EXAMPLE_VALUES: 
            data[col] = EXAMPLE_VALUES[col][:n]
        else:
            ### this will automatically add new column if it changes
            #NA's are because this breaks without sample data
            data[col] = [pd.NA] * n

    df = pd.DataFrame(data)

    ##Enforce the type of data
    df = df.astype(CELLDATA_DTYPES, errors="ignore")

    return df

def coerce_and_validate_celldata_df(df: pd.DataFrame) -> pd.DataFrame:
    #First check if any columns are missing from the input CSV
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"It's missing column: {missing}")
    
    ##slect the column
    df = df[REQUIRED_COLUMNS].copy()

    ##Coerce numeric into datatype and check. Also throw back any errors. 
    for col in NUMERIC_COLUMNS:
        original = df[col]

        #Clean up most common problems (whitespace). Keeps it simple but helpful.
        if original.dtype == "object":
            cleaned = original.astype(str).str.strip()
            #This preserve true missing values (so 'nan' from astype(str) doesn't confuse) can be helpful
            cleaned = cleaned.where(original.notna(), pd.NA)
        else:
            cleaned = original

        numeric = pd.to_numeric(cleaned, errors="coerce")

        #In this case, "bad" means values that were not NaNs became NaN after coercion
        bad_mask = numeric.isna() & original.notna()

        if bad_mask.any():
            bad_rows = df.index[bad_mask].tolist()
            bad_values = original.loc[bad_mask].astype(str)

            n_bad = int(bad_mask.sum())
            examples = bad_values.unique()[:8].tolist()
            example_rows = bad_rows[:8]
            ##This should cover the errors, the errors will be displayd in the shiny app.
            raise ValueError(
                f"Column '{col}' must be numeric, however {n_bad} row(s) could not be converted and are probably not numerical.\n"
                f"Example of a bad value(s): {examples}\n"
                f"Example of a bad row: {example_rows}\n"
                f"Tip: check for text, commas as decimals (such as '12,3' in stead of '12.3'), or stray characters that are not numerical."
            )

        df[col] = numeric.astype("float64")

    #coerce string
    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string")

    return df
