#!/usr/bin/env python
import argparse
import json
import os
import re
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger
from src.data.load_data import read_any
from src.eda.eda import (
    ensure_dir, data_overview, data_dictionary, quality_flags,
    save_dictionary_csv, plot_missing_bar, plot_histograms,
    plot_boxplots, plot_correlation
)

def clean_filename(filename):
    """Clean filename to avoid filesystem issues"""
    # Remove or replace problematic characters
    clean = re.sub(r'[/<>:"|?*]', '_', filename)
    # Limit length to avoid filesystem limits
    if len(clean) > 100:
        clean = clean[:97] + "..."
    # Remove multiple underscores
    clean = re.sub(r'_+', '_', clean)
    return clean.strip('_')

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CRISP-DM · Data Understanding (VS Code + Git)"
    )
    ap.add_argument("--input", required=True, help="Ruta al archivo (.xlsx/.csv)")
    ap.add_argument("--sep", default="auto", help="Separador CSV (auto, ',', ';', '\t', '|')")
    ap.add_argument("--sheet", default=None, help="Hoja Excel (índice o nombre)")
    ap.add_argument("--max-hist", type=int, default=12, help="Máx. columnas numéricas para histogramas")
    ap.add_argument("--max-box", type=int, default=8, help="Máx. columnas numéricas para boxplots")
    return ap.parse_args()

def main() -> None:
    logger = get_logger()
    args = parse_args()

    # Prepare outputs
    ensure_dir("reports")
    ensure_dir("reports/figures")
    ensure_dir("data/interim")

    # Load data
    sheet: Optional[Union[int,str]] = None
    if args.sheet is not None:
        try:
            sheet = int(args.sheet)
        except ValueError:
            sheet = args.sheet

    logger.info(f"Leyendo datos: {args.input}")
    df = read_any(args.input, sep=args.sep, sheet=sheet)

    # Handle case where read_any returns a dictionary (multiple sheets)
    if isinstance(df, dict):
        if sheet is None:
            # If no sheet specified, use the first one
            first_sheet = list(df.keys())[0]
            logger.warning(f"Multiple sheets found. Using first sheet: {first_sheet}")
            df = df[first_sheet]
        else:
            # Use specified sheet
            if sheet in df:
                df = df[sheet]
            else:
                logger.error(f"Sheet '{sheet}' not found. Available sheets: {list(df.keys())}")
                return

    # Save interim copy with normalized column names
    df_cols_norm = df.copy()
    df_cols_norm.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    df_cols_norm.to_csv("data/interim/sample_head.csv", index=False)

    # Overview
    ov = data_overview(df_cols_norm)
    logger.info(f"Dimensiones: {ov['rows']} filas × {ov['cols']} columnas")
    logger.info(f"Primeras columnas: {ov['columns'][:5]}")

    # Dictionary
    dd = data_dictionary(df_cols_norm)
    save_dictionary_csv(dd, "reports/data_dictionary.csv")
    dd.to_csv("data/interim/_data_dictionary_debug.csv", index=False)

    # Quality flags
    flags = quality_flags(df_cols_norm, dd)
    with open("reports/quality_flags.json", "w", encoding="utf-8") as f:
        json.dump(flags, f, ensure_ascii=False, indent=2)
    logger.info(f"Flags de calidad: {flags}")

    # Plots
    logger.info("Generando gráficos...")
    plot_missing_bar(df_cols_norm, "reports/figures/missing_bar.png")
    plot_histograms(df_cols_norm, "reports/figures", max_cols=args.max_hist)
    plot_boxplots(df_cols_norm, "reports/figures", max_cols=args.max_box)
    plot_correlation(df_cols_norm, "reports/figures/corr_matrix.png")

    # Export basic profiling tables
    # Numeric summary
    num = df_cols_norm.select_dtypes(include=[np.number])
    if num.shape[1] > 0:
        num.describe().to_csv("reports/numeric_summary.csv")
    
    # Categorical freq (top 20) - with cleaned filenames
    cat = df_cols_norm.select_dtypes(exclude=[np.number])
    freq_out = []
    for c in cat.columns:
        vc = cat[c].value_counts(dropna=False).head(20)
        # Clean column name for filename
        clean_col_name = clean_filename(c)
        try:
            vc.to_csv(f"reports/{clean_col_name}_top20_value_counts.csv")
            freq_out.append(c)
        except Exception as e:
            logger.warning(f"No se pudo guardar archivo para columna '{c}': {e}")

    logger.info("Listo. Revisa carpeta 'reports' y 'reports/figures'.")

if __name__ == "__main__":
    main()