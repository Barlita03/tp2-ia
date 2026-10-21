"""
Script de Depuracion y Particion Estratificada de Datos (60/20/20)
Trabajo Practico N° 2 - Sistemas Inteligentes - Grupo 9
UTN FRBA - 2026
"""

import os
import csv
import zipfile
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "03_Datos")
ZIP_PATH = os.path.join(ROOT_DIR, "dataset.zip")
DUP_CSV = os.path.join(DATA_DIR, "duplicados_exactos.csv")

LOCAL_DATASET_DIR = r"C:\datasets\mushrooms_10"

TARGET_10_CLASSES = [
    "Agaricus_campestris",
    "Aleuria_aurantia",
    "Amanita_muscaria",
    "Amanita_pantherina",
    "Boletus_edulis",
    "Cantharellus_cibarius",
    "Cortinarius_violaceus",
    "Lactarius_indigo",
    "Pleurotus_ostreatus",
    "Stropharia_aeruginosa"
]

def main():
    print("=" * 60)
    print("Iniciando depuracion y particion de datos (Semilla=42)")
    print("=" * 60)

    # 1. Cargar duplicados para descartar
    df_dup = pd.read_csv(DUP_CSV)
    files_to_discard = set()

    for _, row in df_dup.iterrows():
        archivos = [f.strip() for f in row["archivos"].split(";")]
        if row["distintas_clases"] == True or str(row["distintas_clases"]).lower() == "true":
            # Conflicto inter-clase: descartamos todas las instancias ambiguas
            for f in archivos:
                files_to_discard.add(f)
        else:
            # Duplicados intra-clase: conservamos el primero, descartamos el resto
            for f in archivos[1:]:
                files_to_discard.add(f)

    print(f"Total de instancias duplicadas marcadas para descarte: {len(files_to_discard)}")

    # 2. Leer todas las imagenes validas del ZIP
    valid_entries = []
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            fn = info.filename
            if fn in files_to_discard:
                continue
            parts = fn.split("/")
            if fn.startswith("images/") and len(parts) >= 3:
                cls_name = parts[1]
                valid_entries.append((fn, cls_name))

    print(f"Total de imagenes unicas y limpias conservadas: {len(valid_entries):,}")

    df_clean = pd.DataFrame(valid_entries, columns=["filename", "clase"])
    all_classes_sorted = sorted(df_clean["clase"].unique())
    class_to_id = {cls: i for i, cls in enumerate(all_classes_sorted)}
    df_clean["clase_id"] = df_clean["clase"].map(class_to_id)

    # 3. Particion estratificada 60/20/20 sobre TODO el dataset (277 clases)
    # Primero: 80% train+val, 20% test
    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df_clean["clase"]
    )

    # Segundo: del 80%, 75% es train (60% total) y 25% es val (20% total)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=train_val_df["clase"]
    )

    print("\n--- Particion Completa (277 clases) ---")
    print(f"Train (60%): {len(train_df):,} imagenes")
    print(f"Val   (20%): {len(val_df):,} imagenes")
    print(f"Test  (20%): {len(test_df):,} imagenes")
    print(f"Total:       {len(train_df) + len(val_df) + len(test_df):,} imagenes")

    # Guardar CSVs completos
    train_csv_path = os.path.join(DATA_DIR, "train.csv")
    val_csv_path = os.path.join(DATA_DIR, "val.csv")
    test_csv_path = os.path.join(DATA_DIR, "test.csv")

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    print("CSVs completos guardados: train.csv, val.csv, test.csv")

    # 4. Filtrar y preparar subconjunto representativo de 10 clases
    print("\n--- Subconjunto Representativo de 10 Clases ---")
    c10_to_id = {cls: i for i, cls in enumerate(sorted(TARGET_10_CLASSES))}

    train_10 = train_df[train_df["clase"].isin(TARGET_10_CLASSES)].copy()
    val_10 = val_df[val_df["clase"].isin(TARGET_10_CLASSES)].copy()
    test_10 = test_df[test_df["clase"].isin(TARGET_10_CLASSES)].copy()

    train_10["clase_id"] = train_10["clase"].map(c10_to_id)
    val_10["clase_id"] = val_10["clase"].map(c10_to_id)
    test_10["clase_id"] = test_10["clase"].map(c10_to_id)

    print(f"Train 10 (60%): {len(train_10):,} imagenes")
    print(f"Val   10 (20%): {len(val_10):,} imagenes")
    print(f"Test  10 (20%): {len(test_10):,} imagenes")
    print(f"Total 10:       {len(train_10) + len(val_10) + len(test_10):,} imagenes")

    train_10_path = os.path.join(DATA_DIR, "train_10clases.csv")
    val_10_path = os.path.join(DATA_DIR, "val_10clases.csv")
    test_10_path = os.path.join(DATA_DIR, "test_10clases.csv")

    train_10.to_csv(train_10_path, index=False)
    val_10.to_csv(val_10_path, index=False)
    test_10.to_csv(test_10_path, index=False)
    print("CSVs 10 clases guardados: train_10clases.csv, val_10clases.csv, test_10clases.csv")

    # 5. Extraer imagenes de las 10 clases a C:\datasets\mushrooms_10 fuera de OneDrive
    print(f"\nExtrayendo imagenes de 10 clases a carpeta local {LOCAL_DATASET_DIR}...")
    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

    extracted_count = 0
    all_10_files = set(train_10["filename"]).union(set(val_10["filename"])).union(set(test_10["filename"]))

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for fn in all_10_files:
            target_path = os.path.join(LOCAL_DATASET_DIR, fn.replace("/", os.sep))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "wb") as f_out:
                f_out.write(z.read(fn))
            extracted_count += 1
            if extracted_count % 1000 == 0:
                print(f"  Extraidas {extracted_count}/{len(all_10_files)} imagenes...")

    print(f"Extraccion finalizada: {extracted_count} imagenes en {LOCAL_DATASET_DIR}")
    print("Preparacion de particiones completada con exito.")

if __name__ == "__main__":
    main()
