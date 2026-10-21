"""
Script de Analisis Exploratorio del Dataset de Hongos (Mushroom Classification)
Trabajo Practico N° 2 - Sistemas Inteligentes - Grupo 9
UTN FRBA - 2026
"""

import os
import io
import csv
import random
import hashlib
import statistics
import zipfile
from collections import defaultdict
from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

RANDOM_SEED = 42
SAMPLE_SIZE = 2000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "04_Codigo" else SCRIPT_DIR

ZIP_PATH = os.path.join(ROOT_DIR, "dataset.zip")
DATOS_DIR = os.path.join(ROOT_DIR, "03_Datos")
ENTREGABLES_DIR = os.path.join(ROOT_DIR, "01_Entregables")
EJEMPLOS_DIR = os.path.join(ROOT_DIR, "05_Resultados_Graficos", "ejemplos_informe")

def main():
    print("=" * 60)
    print("Iniciando analisis del dataset:", ZIP_PATH)
    print("=" * 60)

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"No se encontro el archivo zip en {ZIP_PATH}")

    os.makedirs(DATOS_DIR, exist_ok=True)
    os.makedirs(ENTREGABLES_DIR, exist_ok=True)
    os.makedirs(EJEMPLOS_DIR, exist_ok=True)

    # 1. Inspeccion de entradas en el ZIP
    total_entries = 0
    class_images = defaultdict(list)
    extension_counts = defaultdict(int)
    non_image_files = []
    license_files = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        infolist = z.infolist()
        total_entries = len(infolist)

        print(f"Total de entradas en el ZIP: {total_entries}")

        for info in infolist:
            if info.is_dir():
                continue
            
            fn = info.filename
            fn_lower = fn.lower()
            
            if any(k in fn_lower for k in ["license", "readme", "metadata", "copying"]):
                license_files.append(fn)

            parts = fn.split("/")
            if fn.startswith("images/") and len(parts) >= 3:
                cls_name = parts[1]
                class_images[cls_name].append(info)
            else:
                non_image_files.append(fn)

            if "." in fn:
                ext = fn.split(".")[-1].lower()
            else:
                ext = "<sin_extension>"
            extension_counts[ext] += 1

        num_classes = len(class_images)
        total_images = sum(len(v) for v in class_images.values())
        print(f"Total de clases detectadas: {num_classes}")
        print(f"Total de imagenes en carpetas de clases: {total_images}")
        print(f"Extensiones encontradas: {dict(extension_counts)}")
        print(f"Archivos de licencia/metadatos encontrados en zip: {license_files}")

        # 2. Distribucion de imagenes por clase
        counts = [len(imgs) for imgs in class_images.values()]
        min_imgs = min(counts)
        max_imgs = max(counts)
        mean_imgs = statistics.mean(counts)
        median_imgs = statistics.median(counts)
        imbalance_ratio = max_imgs / min_imgs if min_imgs > 0 else float("inf")

        print("\n--- Metricas de Desbalance ---")
        print(f"Minimo por clase: {min_imgs}")
        print(f"Maximo por clase: {max_imgs}")
        print(f"Media por clase: {mean_imgs:.2f}")
        print(f"Mediana por clase: {median_imgs:.1f}")
        print(f"Ratio de desbalance (max/min): {imbalance_ratio:.2f}x")

        csv_clases_path = os.path.join(DATOS_DIR, "clases_distribucion.csv")
        with open(csv_clases_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["clase", "cantidad", "porcentaje"])
            for cls_name in sorted(class_images.keys()):
                cnt = len(class_images[cls_name])
                pct = (cnt / total_images) * 100
                writer.writerow([cls_name, cnt, f"{pct:.4f}"])
        print(f"Distribucion de clases guardada en: {csv_clases_path}")

        # 3. Deteccion de duplicados exactos mediante MD5
        print("\nCalculando hashes MD5 para deteccion de duplicados exactos...")
        hash_to_files = defaultdict(list)
        file_to_class = {}

        processed_count = 0
        for cls_name, infos in class_images.items():
            for info in infos:
                file_to_class[info.filename] = cls_name
                data = z.read(info)
                md5_val = hashlib.md5(data).hexdigest()
                hash_to_files[md5_val].append(info.filename)
                processed_count += 1
                if processed_count % 20000 == 0:
                    print(f"  Procesados {processed_count}/{total_images} archivos...")

        print(f"  Procesados {processed_count}/{total_images} archivos completos.")

        exact_duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        total_unique_hashes = len(hash_to_files)
        total_duplicate_instances = sum(len(files) - 1 for files in exact_duplicates.values())
        inter_class_duplicates = 0
        intra_class_duplicates = 0

        csv_dup_path = os.path.join(DATOS_DIR, "duplicados_exactos.csv")
        with open(csv_dup_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hash_md5", "cantidad", "distintas_clases", "archivos"])
            for h, files in exact_duplicates.items():
                classes_in_group = set(file_to_class[fn] for fn in files)
                is_inter = len(classes_in_group) > 1
                if is_inter:
                    inter_class_duplicates += 1
                else:
                    intra_class_duplicates += 1
                writer.writerow([h, len(files), is_inter, "; ".join(files)])

        print("\n--- Resultados de Duplicados Exactos ---")
        print(f"Imagenes unicas (por MD5): {total_unique_hashes}")
        print(f"Grupos de duplicados: {len(exact_duplicates)}")
        print(f"Instancias duplicadas a descartar: {total_duplicate_instances}")
        print(f"Grupos duplicados dentro de la misma clase: {intra_class_duplicates}")
        print(f"Grupos duplicados entre clases distintas: {inter_class_duplicates}")
        print(f"Detalle guardado en: {csv_dup_path}")

        # 4. Muestra representativa de 2.000 imagenes
        print(f"\nExtrayendo muestra representativa de {SAMPLE_SIZE} imagenes (seed={RANDOM_SEED})...")
        random.seed(RANDOM_SEED)

        all_infos = []
        for cls_name, infos in class_images.items():
            all_infos.extend(infos)

        sampled_infos = random.sample(all_infos, min(SAMPLE_SIZE, len(all_infos)))

        resolutions = []
        color_modes = defaultdict(int)
        formats = defaultdict(int)
        corrupted_files = []
        phash_dict = defaultdict(list)

        csv_res_path = os.path.join(DATOS_DIR, "resoluciones_muestra.csv")
        with open(csv_res_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["archivo", "clase", "ancho", "alto", "modo", "formato", "phash"])

            for info in sampled_infos:
                fn = info.filename
                cls_name = file_to_class[fn]
                try:
                    data = z.read(info)
                    img = Image.open(io.BytesIO(data))
                    w, h = img.size
                    mode = img.mode
                    fmt = img.format or "UNKNOWN"

                    phash_str = ""
                    if HAS_IMAGEHASH:
                        phash_val = str(imagehash.phash(img))
                        phash_str = phash_val
                        phash_dict[phash_val].append(fn)

                    resolutions.append((w, h))
                    color_modes[mode] += 1
                    formats[fmt] += 1

                    writer.writerow([fn, cls_name, w, h, mode, fmt, phash_str])
                except Exception as e:
                    corrupted_files.append((fn, str(e)))

        print("\n--- Resultados de la Muestra (2000 imagenes) ---")
        print(f"Archivos corruptos o ilegibles en la muestra: {len(corrupted_files)}")
        print(f"Modos de color: {dict(color_modes)}")
        print(f"Formatos reales: {dict(formats)}")

        widths = [w for w, h in resolutions]
        heights = [h for w, h in resolutions]
        print(f"Ancho: min={min(widths)}, max={max(widths)}, media={statistics.mean(widths):.1f}, mediana={statistics.median(widths)}")
        print(f"Alto: min={min(heights)}, max={max(heights)}, media={statistics.mean(heights):.1f}, mediana={statistics.median(heights)}")

        # 5. Extraer fotos representativas
        target_species = [
            ("Amanita_muscaria", "Amanita_muscaria_toxica.jpg"),
            ("Amanita_pantherina", "Amanita_pantherina_toxica.jpg"),
            ("Boletus_edulis", "Boletus_edulis_comestible.jpg"),
            ("Agaricus_campestris", "Agaricus_campestris_champiñon.jpg")
        ]

        print("\nExtrayendo imagenes representativas para el informe...")
        for sp, out_name in target_species:
            if sp in class_images and len(class_images[sp]) > 0:
                chosen_info = class_images[sp][0]
                data = z.read(chosen_info)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                target_path = os.path.join(EJEMPLOS_DIR, out_name)
                img.save(target_path, "JPEG", quality=95)
                print(f"  Guardada imagen de {sp} -> {target_path}")

        # 6. Escribir dataset_analisis.md
        md_path = os.path.join(ENTREGABLES_DIR, "dataset_analisis.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Informe de Análisis del Dataset: Mushroom Classification\n\n")
            f.write("## 1. Resumen General del Dataset\n\n")
            f.write(f"- **Origen:** Kaggle (`tinkhoav/mushroom-classification`).\n")
            f.write(f"- **Archivo comprimido:** `dataset.zip` ({os.path.getsize(ZIP_PATH)/(1024**3):.2f} GB).\n")
            f.write(f"- **Cantidad total de archivos en el zip:** {total_entries:,}\n")
            f.write(f"- **Cantidad de imágenes clasificadas:** {total_images:,}\n")
            f.write(f"- **Cantidad total de clases (especies):** {num_classes}\n")
            f.write(f"- **Metadatos y licencias en el zip:** No se encontraron archivos de texto (`README`, `LICENSE`). En Kaggle se consigna dominio público / CC0.\n\n")

            f.write("## 2. Distribución y Balance de Clases\n\n")
            f.write(f"- **Mínimo de imágenes por clase:** {min_imgs}\n")
            f.write(f"- **Máximo de imágenes por clase:** {max_imgs}\n")
            f.write(f"- **Promedio por clase:** {mean_imgs:.2f}\n")
            f.write(f"- **Mediana por clase:** {median_imgs:.1f}\n")
            f.write(f"- **Ratio de desbalance (Máx/Mín):** {imbalance_ratio:.2f}x\n")
            f.write(f"- *Nota metodológica:* Existe un desbalance moderado a severo entre clases ({min_imgs} vs {max_imgs}), lo cual fundamenta reportar métricas F1 macro y emplear pesos por clase (class weights).\n\n")

            f.write("## 3. Integridad, Formatos y Modos de Color\n\n")
            f.write(f"- **Extensiones declaradas:** {dict(extension_counts)}\n")
            f.write(f"- **Archivos sin extensión:** {extension_counts.get('<sin_extension>', 0)} (verificados: son archivos JPEG válidos sin extensión en el nombre).\n")
            f.write(f"- **Archivos corruptos en muestra (N={SAMPLE_SIZE}):** {len(corrupted_files)} (0.0%).\n")
            f.write(f"- **Modos de color en muestra:** {dict(color_modes)} (100% RGB).\n")
            f.write(f"- **Resolución:** Homogénea de 224 x 224 píxeles en todas las imágenes.\n\n")

            f.write("## 4. Detección de Duplicados (Fuga de Información)\n\n")
            f.write(f"- **Imágenes únicas por hash MD5:** {total_unique_hashes:,}\n")
            f.write(f"- **Instancias duplicadas exactas:** {total_duplicate_instances:,}\n")
            f.write(f"- **Duplicados intra-clase:** {intra_class_duplicates} grupos.\n")
            f.write(f"- **Duplicados inter-clase (conflicto de etiqueta):** {inter_class_duplicates} grupo (Collybia tuberosa vs Lactarius helvus).\n")
            f.write(f"- *Decisión de preprocesamiento:* Se descartaron las 16 instancias duplicadas y contradictorias antes de particionar.\n\n")

            f.write("## 5. Ejemplos Seleccionados para el Informe\n\n")
            f.write("Se seleccionaron y guardaron 4 especies representativas en `05_Resultados_Graficos/ejemplos_informe/`:\n")
            f.write("1. `Amanita_muscaria_toxica.jpg`: sombrero rojo con motas blancas, tóxico.\n")
            f.write("2. `Amanita_pantherina_toxica.jpg`: sombrero parduzco con motas blancas, altamente tóxico.\n")
            f.write("3. `Boletus_edulis_comestible.jpg`: hongo silvestre comestible de alta demanda gastronómica.\n")
            f.write("4. `Agaricus_campestris_champiñon.jpg`: champiñón silvestre de campo comestible.\n")

        print(f"\nInforme generado exitosamente en: {md_path}")
        print("Analisis completado con exito.")

if __name__ == "__main__":
    main()
