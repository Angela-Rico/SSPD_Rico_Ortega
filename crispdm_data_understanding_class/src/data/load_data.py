from typing import Union, Optional
import pandas as pd
import os

def read_any(path: str, sep: Optional[str] = "auto", sheet: Union[int, str, None] = None) -> pd.DataFrame:
    """Read CSV/Excel robustly.
    - sep='auto' tries ',', ';', '\t' for CSVs.
    - Excel: uses openpyxl engine; sheet can be index or name.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    lower = path.lower()
    if lower.endswith(('.xls', '.xlsx')):
        return pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
    elif lower.endswith('.csv'):
        if sep == 'auto':
            for candidate in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(path, sep=candidate)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
            # fallback
            return pd.read_csv(path)
        else:
            return pd.read_csv(path, sep=sep)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")