from __future__ import annotations

import pandas as pd


def initialize_metadata_workspace(
    master_df: pd.DataFrame,
    *,
    key_col: str,
) -> pd.DataFrame:
    """
    Create a metadata workspace containing one row per unique key value.

    The master clustering dataset is authoritative for key values.
    No normalization, trimming, case conversion, or fuzzy matching is done.
    """
    if master_df is None or master_df.empty:
        raise ValueError(
            "No clustering dataset is available."
        )

    if key_col not in master_df.columns:
        raise ValueError(
            f"Metadata key column {key_col!r} is not present "
            "in the clustering dataset."
        )

    key_series = master_df[key_col]

    if key_series.isna().any():
        raise ValueError(
            f"Metadata key column {key_col!r} contains missing values."
        )

    # Strings that are literally empty/whitespace are invalid keys,
    # but we deliberately do NOT modify them.
    string_keys = key_series.astype("string")

    bad_key = (
        string_keys.isna()
        | (string_keys.str.len() == 0)
        | string_keys.str.isspace()
    )

    if bad_key.any():
        raise ValueError(
            f"Metadata key column {key_col!r} contains empty values."
        )

    return (
        master_df[[key_col]]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )


def _validate_incoming_metadata_columns(
    incoming_df: pd.DataFrame,
) -> None:
    """
    Validate column names and incoming metadata structure.

    The first column is the matching key supplied by the external CSV.
    """
    if incoming_df is None or incoming_df.empty:
        raise ValueError(
            "The metadata table is empty."
        )

    if len(incoming_df.columns) < 2:
        raise ValueError(
            "The metadata table must contain a matching-key column "
            "and at least one metadata column."
        )

    column_names = list(incoming_df.columns)

    bad_names = [
        c
        for c in column_names
        if str(c).strip() == ""
    ]

    if bad_names:
        raise ValueError(
            "Metadata column names cannot be empty."
        )

    if len(column_names) != len(set(column_names)):
        raise ValueError(
            "Metadata column names must be unique."
        )


def merge_metadata_into_workspace(
    workspace_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    *,
    key_col: str,
    master_columns: list[str],
) -> pd.DataFrame:
    """
    Add metadata columns from an imported table to the
    persistent metadata workspace.

    The imported table must contain a column whose name matches
    the selected PINT metadata key.

    Matching is performed on conservatively normalized string
    values:
    - values are converted to strings
    - leading/trailing whitespace is ignored
    - matching remains case-sensitive
    """

    if incoming_df is None or incoming_df.empty:
        raise ValueError(
            "The metadata table is empty."
        )

    if workspace_df is None or workspace_df.empty:
        raise ValueError(
            "Metadata workspace has not been initialized."
        )

    if key_col not in workspace_df.columns:
        raise ValueError(
            f"Metadata workspace does not contain "
            f"the selected key {key_col!r}."
        )

    incoming = incoming_df.copy()

    # --------------------------------------------------------
    # Clean imported column names
    # --------------------------------------------------------

    incoming.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        for column in incoming.columns
    ]

    if incoming.columns.duplicated().any():
        duplicated = (
            pd.Index(incoming.columns)[
                pd.Index(incoming.columns).duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Metadata table contains duplicate column names: "
            + ", ".join(
                map(str, duplicated)
            )
        )

    # --------------------------------------------------------
    # Find the matching-key column BY NAME
    # --------------------------------------------------------

    if key_col not in incoming.columns:
        detected = ", ".join(
            map(str, incoming.columns)
        )

        raise ValueError(
            f"The metadata table does not contain the "
            f"selected matching column {key_col!r}. "
            f"Detected columns: {detected}"
        )

    incoming_metadata_cols = [
        column
        for column in incoming.columns
        if column != key_col
    ]

    if not incoming_metadata_cols:
        raise ValueError(
            "The metadata table contains the matching key "
            "but no metadata columns."
        )

    # --------------------------------------------------------
    # Normalize matching values conservatively
    # --------------------------------------------------------

    workspaceKeys = (
        workspace_df[key_col]
        .astype("string")
        .str.strip()
    )

    incomingKeys = (
        incoming[key_col]
        .astype("string")
        .str.strip()
    )

    badWorkspaceKeys = (
        workspaceKeys.isna()
        | workspaceKeys.eq("")
    )

    if badWorkspaceKeys.any():
        raise ValueError(
            f"The PINT matching column {key_col!r} "
            "contains missing or empty values."
        )

    badIncomingKeys = (
        incomingKeys.isna()
        | incomingKeys.eq("")
    )

    if badIncomingKeys.any():
        badRows = (
            incoming.index[
                badIncomingKeys
            ]
            .tolist()
        )

        raise ValueError(
            "The metadata matching column contains "
            "missing or empty values at row(s): "
            + ", ".join(
                str(i + 2)
                for i in badRows[:20]
            )
        )

    # Use normalized keys internally for matching.
    workspace = workspace_df.copy()
    workspace["_PINT_METADATA_KEY_"] = (
        workspaceKeys
    )

    incoming["_PINT_METADATA_KEY_"] = (
        incomingKeys
    )

    # --------------------------------------------------------
    # Duplicate key check
    # --------------------------------------------------------

    duplicatedMask = (
        incoming["_PINT_METADATA_KEY_"]
        .duplicated(
            keep=False
        )
    )

    if duplicatedMask.any():
        duplicatedValues = (
            incoming.loc[
                duplicatedMask,
                key_col,
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Metadata matching column {key_col!r} "
            "contains duplicate values: "
            + ", ".join(
                duplicatedValues[:20]
            )
        )

    # --------------------------------------------------------
    # Compare metadata keys with PINT keys
    # --------------------------------------------------------

    workspaceKeyList = (
        workspace["_PINT_METADATA_KEY_"]
        .tolist()
    )

    incomingKeyList = (
        incoming["_PINT_METADATA_KEY_"]
        .tolist()
    )

    workspaceSet = set(
        workspaceKeyList
    )

    incomingSet = set(
        incomingKeyList
    )

    missingFromMetadata = [
        value
        for value in workspaceKeyList
        if value not in incomingSet
    ]

    unknownMetadataKeys = [
        value
        for value in incomingKeyList
        if value not in workspaceSet
    ]

    if (
        missingFromMetadata
        or unknownMetadataKeys
    ):
        parts = [
            f"Metadata matching failed for key "
            f"{key_col!r}."
        ]

        if missingFromMetadata:
            parts.append(
                f"{len(missingFromMetadata):,} PINT "
                "key value(s) were missing from the "
                "metadata table. Examples: "
                + ", ".join(
                    map(
                        str,
                        missingFromMetadata[:10],
                    )
                )
            )

        if unknownMetadataKeys:
            parts.append(
                f"{len(unknownMetadataKeys):,} metadata "
                "key value(s) were not present in PINT. "
                "Examples: "
                + ", ".join(
                    map(
                        str,
                        unknownMetadataKeys[:10],
                    )
                )
            )

        raise ValueError(
            "\n".join(parts)
        )

    # --------------------------------------------------------
    # Protect existing columns
    # --------------------------------------------------------

    existingWorkspaceCols = set(
        workspace_df.columns
    )

    workspaceCollisions = [
        column
        for column in incoming_metadata_cols
        if column in existingWorkspaceCols
    ]

    if workspaceCollisions:
        raise ValueError(
            "Metadata column(s) already exist in the "
            "metadata workspace: "
            + ", ".join(
                workspaceCollisions
            )
        )

    protectedMasterCols = (
        set(master_columns)
        - set(workspace_df.columns)
    )

    masterCollisions = [
        column
        for column in incoming_metadata_cols
        if column in protectedMasterCols
    ]

    if masterCollisions:
        raise ValueError(
            "Metadata column(s) conflict with existing "
            "PINT dataset columns: "
            + ", ".join(
                masterCollisions
            )
        )

    # --------------------------------------------------------
    # Check metadata values
    # --------------------------------------------------------

    incomingValues = incoming[
        incoming_metadata_cols
    ]

    badValueMask = (
        incomingValues.isna()
        |
        incomingValues
        .astype("string")
        .apply(
            lambda col: (
                col.str.strip().eq("")
            )
        )
    )

    if badValueMask.any().any():
        badLocations = []

        for column in incoming_metadata_cols:
            badRows = incoming.loc[
                badValueMask[column],
                key_col,
            ].tolist()

            if badRows:
                badLocations.append(
                    f"{column}: "
                    + ", ".join(
                        map(
                            str,
                            badRows[:20],
                        )
                    )
                )

        raise ValueError(
            "Imported metadata contains empty values.\n"
            + "\n".join(
                badLocations
            )
        )

    # --------------------------------------------------------
    # Reorder metadata to match PINT
    # --------------------------------------------------------

    incoming = (
        incoming
        .set_index(
            "_PINT_METADATA_KEY_"
        )
        .loc[
            workspaceKeyList
        ]
        .reset_index(
            drop=True
        )
    )

    incoming[key_col] = (
        workspace_df[key_col]
        .values
    )

    incoming = incoming[
        [key_col]
        + incoming_metadata_cols
    ]

    # --------------------------------------------------------
    # Merge into workspace
    # --------------------------------------------------------

    merged = workspace_df.merge(
        incoming,
        on=key_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    return merged

def add_empty_metadata_column(
    workspace_df: pd.DataFrame,
    *,
    key_col: str,
    column_name: str,
    master_columns: list[str],
) -> pd.DataFrame:
    """
    Add one empty editable metadata column to the workspace.
    """
    if workspace_df is None or workspace_df.empty:
        raise ValueError(
            "Initialize the metadata workspace first."
        )

    new_name = str(
        column_name or ""
    ).strip()

    if not new_name:
        raise ValueError(
            "Metadata column name cannot be empty."
        )

    if new_name == key_col:
        raise ValueError(
            f"{new_name!r} is already the metadata key."
        )

    if new_name in workspace_df.columns:
        raise ValueError(
            f"Metadata column {new_name!r} already exists."
        )

    # A previously committed metadata column will already be present
    # in both the workspace and master dataset, and has already been
    # caught above. Any other master collision is unsafe.
    if new_name in master_columns:
        raise ValueError(
            f"Column {new_name!r} already exists in the master dataset."
        )

    out = workspace_df.copy()
    out[new_name] = ""

    return out


def validate_metadata_workspace_complete(
    workspace_df: pd.DataFrame,
    *,
    key_col: str,
) -> None:
    """
    Require every metadata value to be filled before committing.

    The key column is excluded because it is owned by the master dataset.
    """
    if workspace_df is None or workspace_df.empty:
        raise ValueError(
            "Metadata workspace is empty."
        )

    if key_col not in workspace_df.columns:
        raise ValueError(
            f"Metadata workspace is missing key column {key_col!r}."
        )

    metadata_cols = [
        c
        for c in workspace_df.columns
        if c != key_col
    ]

    if not metadata_cols:
        raise ValueError(
            "No metadata columns have been added."
        )

    values = workspace_df[
        metadata_cols
    ]

    bad_mask = values.isna()

    string_values = values.astype(
        "string"
    )

    empty_string_mask = string_values.apply(
        lambda col: (
            col.str.len().eq(0)
            | col.str.isspace()
        )
    )

    bad_mask = (
        bad_mask
        | empty_string_mask
    )

    if not bad_mask.any().any():
        return

    parts = [
        "Metadata contains empty values."
    ]

    for col in metadata_cols:
        bad_keys = workspace_df.loc[
            bad_mask[col],
            key_col,
        ].tolist()

        if bad_keys:
            parts.append(
                f"{col}: "
                + ", ".join(
                    map(
                        str,
                        bad_keys[:20],
                    )
                )
            )

    raise ValueError(
        "\n".join(parts)
    )


def append_metadata_columns_to_master(
    master_df: pd.DataFrame,
    workspace_df: pd.DataFrame,
    *,
    key_col: str,
    id_col: str,
    columns_to_add: list[str],
) -> pd.DataFrame:
    """
    Append only newly committed metadata columns to the master dataset.

    Metadata columns are positioned immediately after PINT_Cell_ID.
    Existing master columns retain their relative order.
    """
    if master_df is None or master_df.empty:
        raise ValueError(
            "Master clustering dataset is empty."
        )

    if id_col not in master_df.columns:
        raise ValueError(
            f"{id_col} must be created before metadata can be "
            "added to the master dataset."
        )

    if key_col not in master_df.columns:
        raise ValueError(
            f"Master dataset is missing metadata key {key_col!r}."
        )

    if not columns_to_add:
        return master_df.copy()

    missing_workspace_cols = [
        c
        for c in columns_to_add
        if c not in workspace_df.columns
    ]

    if missing_workspace_cols:
        raise ValueError(
            "Metadata workspace is missing requested column(s): "
            + ", ".join(missing_workspace_cols)
        )

    collisions = [
        c
        for c in columns_to_add
        if c in master_df.columns
    ]

    if collisions:
        raise ValueError(
            "Cannot add metadata column(s) because they already exist "
            "in the master dataset: "
            + ", ".join(collisions)
        )

    metadata_to_add = workspace_df[
        [key_col] + columns_to_add
    ].copy()

    merged = master_df.merge(
        metadata_to_add,
        on=key_col,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if merged[
        columns_to_add
    ].isna().any().any():
        raise ValueError(
            "Metadata merge produced missing values. "
            "The master dataset has changed or the metadata key no "
            "longer matches."
        )

    old_cols = list(
        master_df.columns
    )

    id_index = old_cols.index(
        id_col
    )

    columns_before_and_id = old_cols[
        :id_index + 1
    ]

    columns_after_id = old_cols[
        id_index + 1:
    ]

    final_order = (
        columns_before_and_id
        + columns_to_add
        + columns_after_id
    )

    return merged[
        final_order
    ].copy()