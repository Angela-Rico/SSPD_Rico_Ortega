import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def data_overview(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()}
    }

def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    dd = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes.values],
        "non_null": df.notna().sum().values,
        "nulls": df.isna().sum().values,
        "null_pct": (df.isna().mean().values * 100).round(2),
        "n_unique": df.nunique(dropna=True).values,
    })
    return dd

def quality_flags(df: pd.DataFrame, dd: pd.DataFrame,
                  high_card_threshold: int = 50,
                  const_threshold: float = 0.99) -> Dict[str, Any]:
    flags = {}
    # constant / quasi-constant
    const_like = []
    for c in df.columns:
        vc = df[c].value_counts(dropna=True, normalize=True)
        if len(vc) == 0:
            continue
        if vc.iloc[0] >= const_threshold:
            const_like.append(c)
    flags["constant_like"] = const_like
    # high cardinality categoricals
    high_card = []
    for c in df.select_dtypes(include=['object', 'string']).columns:
        if df[c].nunique(dropna=True) > high_card_threshold:
            high_card.append(c)
    flags["high_cardinality"] = high_card
    # duplicate rows
    flags["duplicate_rows"] = int(df.duplicated().sum())
    # possible id columns (unique key)
    candidate_ids = [c for c in df.columns if df[c].is_unique]
    flags["candidate_ids"] = candidate_ids
    return flags

def save_dictionary_csv(dd: pd.DataFrame, out_path: str) -> None:
    dd.to_csv(out_path, index=False, encoding='utf-8')

def plot_missing_bar(df: pd.DataFrame, out_path: str) -> None:
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure()
    miss.plot(kind='bar')
    plt.title('Porcentaje de valores faltantes por columna')
    plt.ylabel('Proporción')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_histograms(df: pd.DataFrame, out_dir: str, max_cols: int = 12) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for c in num_cols:
        plt.figure()
        df[c].dropna().hist(bins=30)
        plt.title(f'Histograma: {c}')
        plt.xlabel(c)
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{c}.png"))
        plt.close()

def plot_boxplots(df: pd.DataFrame, out_dir: str, max_cols: int = 8) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for c in num_cols:
        plt.figure()
        df[[c]].plot(kind='box')
        plt.title(f'Boxplot: {c}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"box_{c}.png"))
        plt.close()

def plot_correlation(df: pd.DataFrame, out_path: str) -> None:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr(numeric_only=True)
    plt.figure()
    plt.imshow(corr.values, interpolation='nearest')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Matriz de correlaciones')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()