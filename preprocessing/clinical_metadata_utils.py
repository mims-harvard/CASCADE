"""
Shared helpers for per-dataset clinical/donor-level metadata pre-processing (Methods
9.3): integer-encoding categorical columns while preserving missing values, and
exporting per-column value counts (with the encoder's label mapping) to an Excel
workbook for manual QA. Used by the dataset-specific `clinical_metadata_*.py` scripts
(one per dataset actually used in the paper: AUTISM, HLCA, M2, LUCA, HH, SEATTLE).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder


def integer_encode_columns(df, columns):
    """Integer-encode each of `columns` in-place via a per-column LabelEncoder,
    preserving NaNs (which LabelEncoder can't handle directly). Returns the fitted
    encoders keyed by column name, so the original category names can be recovered
    later (`encoder.classes_[int(code)]`)."""
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        nan_mask = df[col].isna()
        non_nan_values = df[col][~nan_mask]

        encoded = le.fit_transform(non_nan_values)
        encoded_series = pd.Series(data=np.nan, index=df.index, dtype="float")
        encoded_series[~nan_mask] = encoded

        df[col] = encoded_series
        label_encoders[col] = le

    print("\nMappings for each categorical column:")
    for col, le in label_encoders.items():
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Mapping for '{col}': {mapping}")

    return label_encoders


def export_value_counts_excel(df, columns, label_encoders, output_path):
    """Write one sheet per column in `columns` (present in `df`) to an Excel
    workbook, with value counts and, for integer-encoded columns, the recovered
    label mapping alongside the codes."""
    output_path = Path(output_path)
    used_sheet_names = set()  # Excel sheet names are case-insensitive for uniqueness
    with pd.ExcelWriter(output_path) as writer:
        for col in columns:
            if col not in df.columns:
                continue
            value_counts_df = df[col].value_counts().reset_index()
            value_counts_df.columns = [col, 'Count']

            if col in label_encoders:
                label_mapping = dict(zip(range(len(label_encoders[col].classes_)), label_encoders[col].classes_))
                value_counts_df[col + '_mapping'] = value_counts_df[col].map(label_mapping)

            sheet_name = re.sub(r'[\\/*?:\[\]]', '_', col)[:31]
            if sheet_name.lower() in used_sheet_names:
                suffix = f"_{sum(1 for s in used_sheet_names if s.startswith(sheet_name.lower()))}"
                sheet_name = sheet_name[:31 - len(suffix)] + suffix
            used_sheet_names.add(sheet_name.lower())

            value_counts_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Value counts saved to {output_path}")
