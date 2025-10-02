#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CRISP-DM · Data Understanding (seguro) con GRÁFICOS por variable y cruces
------------------------------------------------------------------------
- Carga CSV/Excel (sep auto o fijo, hoja por índice/nombre).
- Normaliza nombres de columnas.
- Anonimiza/descarta PII (1–12 y otros sensibles).
- Genera: diccionario de datos, flags de calidad, resúmenes num/cat,
  distribuciones por grupos temáticos y cruces clave.
- Exporta figuras:
    * faltantes (top cols)
    * correlación numérica
    * barras horizontales Top-N por cada variable categórica permitida
    * barras horizontales Top-N para cada cruce definido
Uso:
  python class_data_understanding_secure.py \
      --input data/raw/Enc_SS.xlsx --sep auto --sheet 0
"""

import argparse
import json
import os
import re
import hashlib
import unicodedata
from typing import Optional, Union, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============
# CONFIG BÁSICA
# ============

DROP_COLS = [
    "ObjectID", "GlobalID",
    "5._Para_iniciar,_¿me_puede_decir_nombres_y_apellidos_completos?",
    "6._Tipo_de_Documento", "6.1._Numero_de_documento:",
    "7._Tiene_correo_electronico", "7.1._Proporcione_el_correo",
    "19._Por_favor_indique_su_direccion", "DireccionT",
    "40.1._Numero_de_Factura", "40.2._Cuenta_de_Contrato",
    "Observaciones_del_Encuestador",
    "Creator", "Editor",
]

HASH_COLS = ["1._Nombre_de_encuestador"]

GROUPS: Dict[str, List[str]] = {
    "demografia": [
        "8._Su_edad_se_encuentra_entre",
        "16._Ud._Se_identifica_como",
        "17._Pertenece_a_alguna_de_las_siguientes_etnias",
        "21._¿Cual_es_su_estado_civil_actualmente?",
        "22._¿Cual_es_su_mayor_nivel_de_estudios?",
        "23._Indiqueme_por_favor_si_el_lugar_en_el_que_vive_es:",
        "24._¿Cual_es_su_principal_ocupacion",
        "25._¿Cuantas_personas_actualmente_componen_su_hogar?",
    ],
    "territorial_acceso": [
        "9._Departamento", "10._Municipio", "11._Nombre_Punto_Atencion",
        "20._¿En_que_tipo_de_zona_vive_en_este_momento?",
        "26._¿Cuanto_tiempo_le_toma_desde_su_lugar_de_vivienda_a_este_punto_de_atencion?",
        "27._Aproximadamente,_¿Cuanto_dinero_estima_que_sera_gastado_para_esta_diligencia?",
    ],
    "motivo_gestion": [
        "13._¿Por_que_motivo_viene_usted_hoy_a_este_punto_de_atencion?", "Otro.1",
        "14._Ud._visita_el_punto_de_atencion_¿Como?", "Otro.2",
        "15._¿Coincide_su_lugar_de_residencia_con_el_lugar_donde_esta_siendo_atendido?",
        "28._¿A_que_servicio_refiere_su_visita_al_punto_de_atencion?", "Otro.10",
        "29._¿Que_tipo_de_gestion_llevara_a_cabo?", "Otro.11",
        "30._Su_inconformidad/solicitud_o_peticion_tiene_que_ver_con",
        "30.1_Como_su_respuesta_fue_facturacion,_indique", "Otro.12",
        "30.1_Como_su_respuesta_fue_Prestacion_del_servicio,_indique", "Otro.13",
    ],
    "experiencia_satisfaccion": [
        "31._Como_califica_la_prestacion_del_servicio_por_parte_de_la_empresa", "Otro.14",
        "32._En_este_momento,_¿ud_ya_fue_atendido?",
        "32.1_Elija_la_escala_en_la_cual_se_encuentra_luego_de_ser_atendido",
        "32.1_Elija_la_percepcion_que_tiene_en_este_momento",
        "33._Frente_a_su_motivo_de_visita_hubiera_preferido:", "Otro.15",
        "34.1._¿En_cuanto_tiempo_espera_una_respuesta_a_su_PQR?",
        "34.1_¿En_cuanto_desearia_una_respuesta_a_su_PQR?",
        "35._Si_No_recibe_respuesta_a_su_solicitud_en_tiempo,_¿Que_accion_tomaria?",
        "Buscaria_otra_solucion._¿Cual?",
        "36._Si_Ud._Recibe_una_respuesta_con_la_cual_NO_ESTA_DE_ACUERDO", "Otro.16",
        "37._¿La_empresa_prestadora_le_explico_que_puede_pedir_revision/otra_autoridad?", "Otro.17",
        "38._Sobre_la_SSPD_usted:", "Otro.18",
        "39._¿Que_tan_satisfecho_esta_con_la_atencion_brindada_en_este_punto?",
    ],
}

CROSSES: Dict[str, List[str]] = {
    "municipio_x_punto": ["10._Municipio", "11._Nombre_Punto_Atencion"],
    "motivo_x_servicio": ["13._¿Por_que_motivo_viene_usted_hoy_a_este_punto_de_atencion?",
                          "28._¿A_que_servicio_refiere_su_visita_al_punto_de_atencion?"],
    "calificacion_x_empresa": ["31._Como_califica_la_prestacion_del_servicio_por_parte_de_la_empresa",
                               "12._Empresa"],
    "espera_vs_desea": ["34.1._¿En_cuanto_tiempo_espera_una_respuesta_a_su_PQR?",
                        "34.1_¿En_cuanto_desearia_una_respuesta_a_su_PQR?"],
}


# ============
# UTILIDADES
# ============

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(ch))


def normalize_text(s: str) -> str:
    s = _strip_accents(str(s)).strip()
    s = re.sub(r"\s+", " ", s)
    return s


_EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
_PHONE = re.compile(r'\b(?:\+?\d[\s-]?){8,}\b')


def scrub_free_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = _EMAIL.sub("[EMAIL]", s)
    s = _PHONE.sub("[TEL]", s)
    return normalize_text(s)


def hash_value(val: str, salt: str = "sspd_salt_v1") -> str:
    if pd.isna(val) or str(val).strip() == "":
        return ""
    return hashlib.sha256((salt + str(val)).encode("utf-8")).hexdigest()[:12]


def read_any(path: str, sep: Optional[str] = "auto",
             sheet: Union[int, str, None] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    lower = path.lower()
    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if lower.endswith(".csv"):
        if sep == "auto":
            for candidate in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(path, sep=candidate)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
            return pd.read_csv(path)
        return pd.read_csv(path, sep=sep)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_cols = []
    for c in out.columns:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", " ", c2)
        c2 = c2.replace(" ", "_")
        new_cols.append(c2)
    out.columns = new_cols
    return out


def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in HASH_COLS:
        if c in out.columns:
            out[c] = out[c].map(lambda v: hash_value(v))
    for c in out.select_dtypes(include=["object", "string"]).columns:
        if c in DROP_COLS:
            continue
        if c.startswith("Otro") or "Otro" in c or "Observaciones" in c:
            out[c] = out[c].map(scrub_free_text)
        else:
            out[c] = out[c].map(normalize_text)
    keep = [c for c in out.columns if c not in DROP_COLS]
    return out[keep]


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:150]


# ============
# PERFILAMIENTO
# ============

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


def quality_flags(df: pd.DataFrame, high_card_threshold: int = 50,
                  const_threshold: float = 0.99) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    const_like = []
    for c in df.columns:
        vc = df[c].value_counts(dropna=True, normalize=True)
        if len(vc) and vc.iloc[0] >= const_threshold:
            const_like.append(c)
    flags["constant_like"] = const_like
    high_card = []
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if df[c].nunique(dropna=True) > high_card_threshold:
            high_card.append(c)
    flags["high_cardinality"] = high_card
    flags["duplicate_rows"] = int(df.duplicated().sum())
    candidate_ids = [c for c in df.columns if df[c].is_unique]
    flags["candidate_ids"] = candidate_ids
    return flags


# ============
# GRÁFICOS
# ============

def plot_missing_bar(df: pd.DataFrame, out_path: str, top: Optional[int] = None) -> None:
    miss = df.isna().mean().sort_values(ascending=False)
    if top:
        miss = miss.head(top)
    plt.figure()
    miss.plot(kind="bar")
    plt.title("Porcentaje de valores faltantes por columna")
    plt.ylabel("Proporción")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correlation(df: pd.DataFrame, out_path: str) -> None:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr(numeric_only=True)
    plt.figure()
    plt.imshow(corr.values, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Matriz de correlaciones")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def bar_top_counts(s: pd.Series, out_path: str, top: int = 20, title: str = "") -> None:
    vc = s.value_counts(dropna=False).head(top)[::-1]
    plt.figure()
    vc.plot(kind="barh")
    plt.title(title or (s.name or "Top categorías"))
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def bar_top_pairs(df: pd.DataFrame, cols: List[str], out_path: str, top: int = 30, title: str = "") -> None:
    ct = (
        df.groupby(cols).size().reset_index(name="count")
          .sort_values("count", ascending=False)
          .head(top)
    )
    labels = ct[cols[0]].astype(str) + " | " + ct[cols[1]].astype(str)
    counts = ct["count"].values
    plt.figure()
    y = np.arange(len(counts))
    plt.barh(y, counts)
    plt.yticks(y, labels)
    plt.title(title or f"Top pares: {cols[0]} × {cols[1]}")
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============
# IO AUXILIARES
# ============

def vc_to_csv(s: pd.Series, out: str, top: Optional[int] = None) -> None:
    vc = s.value_counts(dropna=False)
    if top:
        vc = vc.head(top)
    vc.to_csv(out, header=["count"])


# ============
# CLI
# ============

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CRISP-DM · Data Understanding (seguro) + figuras")
    ap.add_argument("--input", required=True, help="Ruta al archivo (.xlsx/.csv)")
    ap.add_argument("--sep", default="auto", help="Separador CSV (auto, ',', ';', '\\t', '|')")
    ap.add_argument("--sheet", default=None, help="Hoja Excel (índice o nombre)")
    ap.add_argument("--outdir", default="reports", help="Carpeta de salida (reportes)")
    ap.add_argument("--max_hist", type=int, default=12, help="Máx. num-cols para hist/box (solo numéricas)")
    ap.add_argument("--top", type=int, default=30, help="Top-N para tablas y barras categóricas")
    ap.add_argument("--high_card_threshold", type=int, default=50, help="Umbral alta cardinalidad")
    ap.add_argument("--const_threshold", type=float, default=0.99, help="Umbral cuasi-constante")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Carpetas
    fig_dir = os.path.join(args.outdir, "figures")
    fig_groups_dir = os.path.join(fig_dir, "groups")
    fig_cross_dir = os.path.join(fig_dir, "cross")
    ensure_dir(args.outdir); ensure_dir(fig_dir); ensure_dir(fig_groups_dir)
    ensure_dir(fig_cross_dir); ensure_dir("data/interim")
    ensure_dir(os.path.join(args.outdir, "groups")); ensure_dir(os.path.join(args.outdir, "cross"))

    # 2) Cargar
    sheet: Optional[Union[int, str]] = None
    if args.sheet is not None:
        try:
            sheet = int(args.sheet)
        except ValueError:
            sheet = args.sheet
    df_raw = read_any(args.input, sep=args.sep, sheet=sheet)
    df = normalize_columns(df_raw)

    # 3) Anonimización / segura
    df_safe = safe_copy(df)
    df_safe.to_csv("data/interim/sample_head.csv", index=False)

    # 4) Diccionario / Flags
    dd = data_dictionary(df_safe)
    dd.to_csv(os.path.join(args.outdir, "data_dictionary.csv"), index=False, encoding="utf-8")
    flags = quality_flags(df_safe, high_card_threshold=args.high_card_threshold,
                          const_threshold=args.const_threshold)
    with open(os.path.join(args.outdir, "quality_flags.json"), "w", encoding="utf-8") as f:
        json.dump(flags, f, ensure_ascii=False, indent=2)

    # 5) Resúmenes básicos
    num = df_safe.select_dtypes(include=[np.number])
    if num.shape[1] > 0:
        num.describe().to_csv(os.path.join(args.outdir, "numeric_summary.csv"))
    cat = df_safe.select_dtypes(exclude=[np.number])

    # Tablas top-N para TODAS las categóricas seguras
    for c in cat.columns:
        out_csv = os.path.join(args.outdir, f"{safe_filename(c)}_top_value_counts.csv")
        vc_to_csv(df_safe[c], out_csv, top=args.top)

    # 6) Figuras generales
    plot_missing_bar(df_safe, os.path.join(fig_dir, "missing_bar.png"), top=50)
    plot_correlation(df_safe, os.path.join(fig_dir, "corr_matrix.png"))

    # Barras horizontales para campos clave (si existen)
    key_fields = {
        "edad": "8._Su_edad_se_encuentra_entre",
        "departamento": "9._Departamento",
        "municipio": "10._Municipio",
        "punto_atencion": "11._Nombre_Punto_Atencion",
    }
    for name, col in key_fields.items():
        if col in df_safe.columns:
            bar_top_counts(
                df_safe[col],
                os.path.join(fig_dir, f"top_{name}.png"),
                top=args.top,
                title=f"Top {name.replace('_',' ').title()}",
            )

    # 7) Figuras por cada variable de CADA GRUPO
    for gname, cols in GROUPS.items():
        g_csv_dir = os.path.join(args.outdir, "groups", gname)
        g_fig_dir = os.path.join(fig_groups_dir, gname)
        ensure_dir(g_csv_dir); ensure_dir(g_fig_dir)
        for c in cols:
            if c in df_safe.columns:
                # tabla
                vc_to_csv(df_safe[c], os.path.join(g_csv_dir, f"{safe_filename(c)}_value_counts.csv"), top=args.top)
                # figura
                bar_top_counts(
                    df_safe[c],
                    os.path.join(g_fig_dir, f"{safe_filename(c)}.png"),
                    top=args.top,
                    title=f"{gname} · {c}"
                )

    # 8) Cruces clave (CSV + figura Top-N pares)
    for cname, pair in CROSSES.items():
        cols_exist = [c for c in pair if c in df_safe.columns]
        if len(cols_exist) == 2:
            # CSV completo ordenado
            ct = (
                df_safe.groupby(cols_exist).size().reset_index(name="count")
                  .sort_values("count", ascending=False)
            )
            ct.to_csv(os.path.join(args.outdir, "cross", f"{safe_filename(cname)}.csv"), index=False)
            # Figura con Top-N pares
            bar_top_pairs(
                df_safe, cols_exist,
                os.path.join(fig_cross_dir, f"{safe_filename(cname)}_top_pairs.png"),
                top=args.top,
                title=f"Top pares · {cols_exist[0]} × {cols_exist[1]}"
            )

    # 9) Mensaje final
    ov = data_overview(df_safe)
    print(f"[OK] Data Understanding seguro + figuras completado.")
    print(f"    Filas x Columnas: {ov['rows']} x {ov['cols']}")
    print(f"    Reportes: {os.path.abspath(args.outdir)}")
    print(f"    Figuras: {os.path.abspath(fig_dir)}")


if __name__ == "__main__":
    main()