# TP2 - Sistemas Inteligentes - Grupo 9
**Materia:** Inteligencia Artificial — UTN FRBA (2026 2C)  
**Tema:** Clasificación de Especies de Hongos a partir de Imágenes con Redes Neuronales Artificiales (MLP vs. CNN 1 Capa vs. CNN 2 Capas)

---

## 📁 Estructura del Proyecto

El repositorio está organizado en carpetas numeradas para que encuentres fácilmente cada componente:

```text
TP2/
│
├── 01_Entregables/                 # Documentos finales listos para presentar o revisar
│   ├── Entrega Final - TP°2 - Sistemas Inteligentes - Grupo 9 - v2.docx
│   ├── Entrega Parcial - TP°2 - Sistemas Inteligentes - Grupo 9 - Hongos - v2.docx
│   └── dataset_analisis.md
│
├── 02_Documentos_Originales/       # Enunciado, plantillas y revisiones originales (intactos)
│   ├── consigna.pdf
│   ├── Entrega Parcial - TP°2 - Sistemas Inteligentes - Grupo 9 - Hongos.docx
│   ├── Entrega Final - TP°2 - Sistemas Inteligentes - Grupo 9.docx
│   └── Revision TP2 - Avances del compañero.docx
│
├── 03_Datos/                       # CSVs de partición, balance y duplicados
│   ├── dataset.txt                 # Enlace al dataset original en Kaggle
│   ├── clases_distribucion.csv     # Cantidad de fotos por especie (277 clases)
│   ├── duplicados_exactos.csv      # 15 grupos duplicados detectados por MD5
│   ├── resoluciones_muestra.csv    # Resoluciones de muestra (224x224 RGB)
│   ├── train.csv, val.csv, test.csv               # Split 60/20/20 del dataset completo
│   └── train_10clases.csv, val_10clases.csv, test_10clases.csv  # Split benchmark 10 clases
│
├── 04_Codigo/                      # Scripts reproducibles en Python (PyTorch)
│   ├── analisis_dataset.py         # Análisis exploratorio en streaming del zip
│   ├── preparar_datos.py           # Limpieza de duplicados y generación de particiones
│   ├── entrenar_modelos.py         # Entrenamiento de los 5 experimentos (MLP, CNN1, CNN2)
│   ├── generar_parcial_v2.py       # Ensamblador automático de la Entrega Parcial v2 (.docx)
│   └── generar_entrega_final_v2.py # Ensamblador de la Entrega Final v2 (.docx)
│
├── 05_Resultados_Graficos/         # Curvas de error, matrices de confusión y métricas
│   ├── resumen_entrenamientos.csv  # Tabla comparativa de todos los experimentos
│   ├── ejemplos_informe/           # Fotos representativas de especies para los informes
│   └── resultados_entrenamientos/  # Gráficos de Loss/Acc y matrices de confusión (Train/Test)
│
├── dataset.zip                     # Archivo comprimido original de Kaggle (2,7 GB)
└── README.md                       # Esta guía
```

---

## 🚀 Modelos Evaluados y Resultados Principales

Todos los modelos fueron entrenados con **PyTorch 2.11 + CUDA (GPU RTX 3060 12GB)** bajo semilla fija (`seed=42`), función de costo **Cross-Entropy ponderada por pesos inversos a la frecuencia de clase** y partición estratificada **60% entrenamiento / 20% validación / 20% prueba**:

| ID | Modelo | Resolución | Optimizador / Tasa | Parámetros | Test Accuracy | Test F1 Macro | Estado |
|:---|:---|:---:|:---:|---:|---:|---:|:---:|
| **Exp 1** | **MLP Base** (2 capas: 256, 128) | 64x64 | Adam ($\text{lr}=10^{-3}$) | 3.181.706 | 37,08% | 36,35% | Convergió |
| **Exp 2** | **CNN 1 Capa** (32 filtros) | 128x128 | Adam ($\text{lr}=10^{-3}$) | 4.200.778 | 36,01% | 30,62% | Cuello de botella |
| **Exp 3** | **CNN 2 Capas** (32 $\to$ 64 filtros) | 128x128 | Adam ($\text{lr}=10^{-3}$) | 2.122.954 | 51,00% | 49,34% | Gran salto cualitativo |
| **Exp 4** | **CNN 2 Capas** (Comparativa SGD) | 128x128 | SGD+Momentum ($\text{lr}=5\cdot10^{-3}$) | 2.122.954 | 45,38% | 43,07% | Convergencia más suave |
| **Exp 5** | **CNN 2 Capas** (Ajuste Fino) | 128x128 | Adam ($\text{lr}=5\cdot10^{-4}$, batch=32) | 2.122.954 | **62,78%** | **62,29%** | **Mejor Modelo** |

---

## 🛠️ Cómo Reproducir el Proyecto

Los scripts están configurados para correr desde la terminal en el entorno virtual (`C:\venvs\tp2`):

```powershell
# 1. Análisis exploratorio
python 04_Codigo\analisis_dataset.py

# 2. Limpieza de duplicados y generación de particiones CSV
python 04_Codigo\preparar_datos.py

# 3. Entrenamiento y generación de gráficos de Loss y matrices
python 04_Codigo\entrenar_modelos.py

# 4. Generación de los documentos Word (.docx)
python 04_Codigo\generar_parcial_v2.py
python 04_Codigo\generar_entrega_final_v2.py
```

---

## 📄 Documentos Listos para Entrega
Los archivos terminados se ubican en `01_Entregables/`:
- `Entrega Parcial - TP°2 - Sistemas Inteligentes - Grupo 9 - Hongos - v2.docx`: Contiene los placeholders completados con números reales, corrección sobre filtros crecientes y aclaración respecto al Anexo de la cátedra.
- `Entrega Final - TP°2 - Sistemas Inteligentes - Grupo 9 - v2.docx`: Informe completo con Introducción, Metodología, Resultados experimentales (tablas y gráficos incrustados), Discusión exhaustiva de las 3 preguntas de la cátedra, Conclusiones y Referencias bibliográficas en formato IEEE.
