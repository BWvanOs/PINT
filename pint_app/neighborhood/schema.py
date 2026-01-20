import pandas as pd

REQUIRED_COLUMNS = [
    "CellID", "X", "Y", "ClusterName", "SampleNumber", "Condition"
]

def example_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CellID": [1, 2, 3, 4, 5, 6],
            "X_Location":      [120.5, 150.2, 180.1,  45.0,  60.3,  90.7],
            "Y_Location":      [ 80.2,  95.1, 110.4, 210.0, 195.6, 205.2],
            "ClusterName": ["T cell", "T cell", "Myeloid", "Stroma", "Stroma", " cell"],
            "SampleNumber": [1, 1, 1, 2, 2, 2],
            "Condition": ["Ctrl", "Ctrl", "Ctrl", "Treat", "Treat", "Treat"],
        }
    )
