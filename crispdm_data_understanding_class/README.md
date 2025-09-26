# CRISP‑DM · Fase: **Data Understanding** (clase en VS Code + Git)

Este repositorio está diseñado para una clase de **2 horas** de *Python for Data Mining* enfocada en la fase **Data Understanding** de CRISP‑DM, trabajando **desde Visual Studio Code** y **con GitHub**.

## Objetivos didácticos
1. Entender cómo una **estructura de proyecto** clara acelera el ciclo CRISP‑DM.
2. Practicar **versionamiento** (Git) desde VS Code.
3. Ejecutar un pipeline reproducible de **carga**, **perfilamiento básico** y **reporte** de datos.

## Estructura
```
crispdm_data_understanding_class/
├─ data/
│  ├─ raw/                # Datos originales (solo lectura)
│  ├─ interim/            # Datos temporales/limpios
│  └─ processed/          # Datos listos para modelado
├─ notebooks/             # (opcional) notebooks de exploración
├─ reports/
│  ├─ figures/            # Gráficos exportados
│  └─ data_dictionary.csv # Diccionario de datos generado
├─ src/
│  ├─ data/
│  │  └─ load_data.py     # Lectura robusta de datos
│  ├─ eda/
│  │  └─ eda.py           # Funciones de Data Understanding
│  └─ utils/
│     └─ logging_utils.py # Logger consistente
├─ .gitignore
├─ requirements.txt
├─ README.md
└─ class_data_understanding.py  # Script principal CLI
```

> Se incluye un dataset de ejemplo: `data/raw/Enc_SS.xlsx`

## Preparación (5-10 min)
1. **Crear repo en GitHub** (privado o público).
2. En **VS Code**: `Ctrl+Shift+P` → *Git: Clone* → pega la URL del repo.
3. En terminal del proyecto:
   ```bash
   python -m venv .venv
   # Activar:
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Verifica que está el archivo de ejemplo en `data/raw/Enc_SS.xlsx`.

## Uso (clase práctica, ~60-75 min)
Ejecuta el *pipeline* de Data Understanding sobre el archivo de ejemplo:
```bash
python class_data_understanding.py --input data/raw/Enc_SS.xlsx --sep auto --sheet 0
```
- Genera:
  - `reports/data_dictionary.csv`
  - `reports/quality_flags.json`
  - Gráficos en `reports/figures/`
  - CSVs auxiliares en `data/interim/`

### Otros ejemplos
- CSV con separador `;`:
  ```bash
  python class_data_understanding.py --input data/raw/datos.csv --sep ";"
  ```
- Excel con hoja por nombre:
  ```bash
  python class_data_understanding.py --input data/raw/otra.xlsx --sheet "Hoja1"
  ```

## Guion sugerido para 2 horas
**0-15 min** · *Setup & Git*
- Clonar repo, crear rama `feature/eda-inicial`, primer commit.
- Explicar `.gitignore`, `requirements.txt`, estructura de carpetas.

**15-75 min** · *Data Understanding hands-on*
- Cargar datos: `src/data/load_data.py`.
- Inspección: tamaño, tipos, nulos, cardinalidades, duplicados.
- Diccionario de datos y *quality flags*.
- Gráficos: faltantes, histogramas, boxplots, correlación.

**75-100 min** · *Buenas prácticas*
- Discute decisiones (constantes, alta cardinalidad, codificación).
- Registrar *issues* y *TODOs* en el repo.

**100-120 min** · *GitHub Flow*
- Pull request → code review (en parejas) → merge a `main`.
- Cierra con *tag* `v0.1-data-understanding`.

---

> Tip: Activa la extensión **Jupyter** en VS Code para correr celdas tipo notebook en archivos `.py` (# %%).