"""
Script de Generacion de la Entrega Final - v2 en formato Word (.docx)
Trabajo Practico N° 2 - Sistemas Inteligentes - Grupo 9
UTN FRBA - 2026
"""

import os
import shutil
import pandas as pd
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DOCS_ORIG_DIR = os.path.join(ROOT_DIR, "02_Documentos_Originales")
ENTREGABLES_DIR = os.path.join(ROOT_DIR, "01_Entregables")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "05_Resultados_Graficos")

SRC_DOCX = os.path.join(DOCS_ORIG_DIR, "Entrega Final - TP°2 - Sistemas Inteligentes - Grupo 9.docx")
DST_DOCX = os.path.join(ENTREGABLES_DIR, "Entrega Final - TP°2 - Sistemas Inteligentes - Grupo 9 - v2.docx")

RESULTS_DIR = os.path.join(OUTPUTS_DIR, "resultados_entrenamientos")
EJEMPLOS_DIR = os.path.join(OUTPUTS_DIR, "ejemplos_informe")
SUMMARY_CSV = os.path.join(OUTPUTS_DIR, "resumen_entrenamientos.csv")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def format_table(table, col_widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(table.rows):
        # Evitar partir fila entre paginas
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        is_header = (r_idx == 0)
        if is_header:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        for c_idx, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if col_widths and c_idx < len(col_widths):
                cell.width = col_widths[c_idx]
            if is_header:
                set_cell_background(cell, "1F497D")
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(8.5)
            else:
                bg = "F2F5F8" if r_idx % 2 == 1 else "FFFFFF"
                set_cell_background(cell, bg)
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8.0)

def add_heading_styled(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    return h

def main():
    print("=" * 60)
    print("Generando Entrega Final - v2...")
    print("=" * 60)

    # Leemos la plantilla base para conservar portada y metadatos
    doc = docx.Document(SRC_DOCX)

    # Identificamos el punto de corte (primer parrafo de Resumen)
    # Conservamos los parrafos de portada (0 a 27)
    cutoff_idx = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("Resumen."):
            cutoff_idx = i
            break

    if cutoff_idx is not None:
        # Borramos los parrafos plantilla desde el Resumen en adelante
        p_elements = [p._p for p in doc.paragraphs[cutoff_idx:]]
        for p_elem in p_elements:
            p_elem.getparent().remove(p_elem)

    # -------------------------------------------------------------
    # RESUMEN Y PALABRAS CLAVES
    # -------------------------------------------------------------
    p_res = doc.add_paragraph()
    p_res.paragraph_format.space_before = Pt(12)
    p_res.paragraph_format.space_after = Pt(6)
    p_res.paragraph_format.line_spacing = 1.15
    r_res_bold = p_res.add_run("Resumen. ")
    r_res_bold.font.bold = True
    r_res_bold.font.size = Pt(10)
    r_res_text = p_res.add_run(
        "Este trabajo presenta la concepción, implementación y evaluación experimental comparativa de Sistemas "
        "Inteligentes basados en Redes Neuronales Artificiales (RNA) aplicados al problema de visión por computadora "
        "para la clasificación taxonómica de especies de hongos a partir de fotografías digitales. El conjunto de datos base "
        "procede del repositorio público de Kaggle 'Mushroom Classification' (tinkhoav), conformado originalmente por 83.729 "
        "imágenes distribuidas en 277 clases botánicas. Se implementó una rigurosa auditoría de integridad que detectó y depuró "
        "16 instancias duplicadas exactas (identificando un caso crítico de contaminación inter-clase en el dataset de origen) "
        "y caracterizó un desbalance muestral con un ratio de 18,50x. Se adoptó una partición estratificada 60/20/20 (entrenamiento, "
        "validación y prueba) con semilla fija reproducible (seed=42) y aumento de datos condicionado únicamente al entrenamiento "
        "(sin volteo vertical). Para resolver el problema y contrastar los paradigmas teóricos vistos en clase, se construyeron y "
        "evaluaron tres arquitecturas principales: (1) Perceptrón Multicapa (MLP) con entrenamiento Backpropagation sobre imágenes "
        "aplanadas a 64x64x3 (3.180.170 parámetros), (2) Red Neuronal Convolucional (CNN) de una etapa convolucional de 32 filtros "
        "sobre 128x128x3 (4.196.618 parámetros), y (3) Red Neuronal Convolucional de dos etapas convolucionales con filtros crecientes "
        "(32 -> 64) y agrupamiento espacial Max Pooling (2.117.962 parámetros). Se realizaron además corridas experimentales contrastando "
        "optimizadores (Adam vs. SGD con Momentum) y ajustes finos de hiperparámetros. Los modelos fueron evaluados exhaustivamente mediante "
        "curvas de convergencia de pérdida de entropía cruzada, matrices de confusión en entrenamiento y prueba, exactitud global y "
        "métricas de Precisión, Recuperación (Recall) y F1-score macro por especie. La CNN de 2 capas optimizada alcanzó una exactitud "
        "de 62,78% y un F1 macro de 62,29%, superando categóricamente al MLP (37,08%) y a la CNN de 1 capa (36,01%). Los resultados "
        "corroboran la teoría de extracción jerárquica de características visuales, la invariancia espacial y la mitigación de la explosión "
        "combinatoria de parámetros en problemas de Computer Vision."
    )
    r_res_text.font.size = Pt(10)

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(4)
    p_kw.paragraph_format.space_after = Pt(16)
    r_kw_bold = p_kw.add_run("Palabras Claves: ")
    r_kw_bold.font.bold = True
    r_kw_bold.font.size = Pt(9.5)
    r_kw_text = p_kw.add_run(
        "sistemas inteligentes, redes neuronales convolucionales (CNN), perceptrón multicapa (MLP), Backpropagation, "
        "clasificación de imágenes, visión por computadora, Softmax, entropía cruzada, data augmentation, desbalance de clases, PyTorch."
    )
    r_kw_text.font.size = Pt(9.5)
    r_kw_text.font.italic = True

    # -------------------------------------------------------------
    # 1. INTRODUCCIÓN
    # -------------------------------------------------------------
    add_heading_styled(doc, "1. Introducción", level=1)

    doc.add_paragraph(
        "El problema abordado en este trabajo práctico consiste en el procesamiento y clasificación automática de imágenes digitales "
        "de hongos para identificar con precisión a qué especie botánica pertenece el ejemplar fotografiado. La identificación micológica "
        "a simple vista es una tarea notoriamente ardua debido a la enorme diversidad del reino Fungi, la alta variabilidad morfológica "
        "dentro de una misma especie y la existencia de numerosas especies 'dobles' con sutiles diferencias visuales. La automatización "
        "de este proceso mediante visión por computadora y Redes Neuronales Artificiales reviste una gran importancia práctica y científica, "
        "estructurada en torno a cuatro ejes fundamentales:"
    )

    doc.add_paragraph(
        "• Seguridad alimentaria y salud pública: La confusión entre especies silvestres comestibles y especies altamente tóxicas o "
        "mortales (como las pertenecientes al género Amanita) constituye una causa recurrente de intoxicaciones graves y letales en todo "
        "el mundo. Un sistema inteligente de clasificación visual brinda una herramienta de orientación y alerta temprana capaz de reducir "
        "el error humano en aficionados y recolectores silvestres.\n"
        "• Micología y preservación de la biodiversidad: La catalogación taxonómica en campañas de relevamiento biológico requiere "
        "inspeccionar cientos de especímenes. La asistencia computacional acelera notablemente la labor de biólogos, ecólogos y naturalistas en campo.\n"
        "• Educación y ecoturismo: Facilita el desarrollo de aplicaciones móviles interactivas para senderistas y estudiantes, promoviendo el "
        "conocimiento botánico responsable y la valoración de los ecosistemas forestales.\n"
        "• Control de calidad agrícola e industrial: Apoya la clasificación y control de pureza varietal en la producción comercial de hongos "
        "cultivados (como champiñones, gírgolas y portobellos), garantizando la ausencia de especímenes espurios en las cadenas de distribución."
    )

    doc.add_paragraph(
        "Aclaración de seguridad: El sistema desarrollado se plantea como una herramienta de apoyo y soporte orientativo a la decisión "
        "humana, y de ninguna manera como un mecanismo autónomo infalible para autorizar el consumo gastronómico de un ejemplar silvestre, "
        "dado que las consecuencias de un falso negativo en toxicidad son potencialmente letales."
    )

    # Aclaracion crucial sobre el Anexo de la catedra
    p_anexo = doc.add_paragraph()
    p_anexo.paragraph_format.space_before = Pt(8)
    p_anexo.paragraph_format.space_after = Pt(8)
    r_anexo_title = p_anexo.add_run("Diferenciación fundamental respecto del Anexo de la cátedra: ")
    r_anexo_title.font.bold = True
    p_anexo.add_run(
        "Se remarca con énfasis que el problema aquí resuelto difiere taxativamente del ítem consignado en el Anexo de la consigna "
        "('Predecir si el hongo es venenoso o no a partir de las características visibles'). Aquel problema histórico refiere a una "
        "clasificación binaria clásica (comestible vs. venenoso) entrenada sobre atributos tabulares categóricos predefinidos por micólogos "
        "en laboratorio (el reconocido dataset Mushroom de UCI Machine Learning Repository, con 22 variables discretas como forma de sombrero, "
        "olor, color de esporas, tipo de velo, etc.). En contraposición absoluta, el presente proyecto aborda una tarea genuina de "
        "Visión por Computadora (Computer Vision) multiclase sobre fotografías no estructuradas de especímenes silvestres capturados "
        "en su hábitat natural, donde la red debe aprender de forma autónoma representaciones jerárquicas espaciales a partir de matrices "
        "de píxeles crudos (RGB), enfrentando ruidos de fondo, oclusiones, variabilidad de luz y perspectiva."
    )

    doc.add_paragraph(
        "Características distintivas del dominio:\n"
        "1. Alta variabilidad intraclase: Ejemplares de una misma especie exhiben variaciones drásticas de coloración, tamaño, textura y "
        "morfología del sombrero según la etapa biológica de maduración, el nivel de humedad ambiental y el sustrato de desarrollo.\n"
        "2. Alta similitud interclase (Fine-Grained Classification): Distintas especies comparten siluetas y tonalidades casi indistinguibles, "
        "haciendo que la tarea requiera detectar detalles texturales muy finos (motas, patrón de láminas, textura del anillo del pie).\n"
        "3. Heterogeneidad en condiciones de captura: Las imágenes presentan fondos complejos y variados (hojarasca, corteza, pastizales, "
        "tierra húmeda), sombras cambiantes, iluminación solar directa o difusa y distintos ángulos de toma.\n"
        "4. Complejidad multiclase: Al abarcar múltiples especies botánicas, el problema supera holgadamente a una clasificación binaria y "
        "presenta un desbalance inherente en la disponibilidad de fotografías por especie en la naturaleza."
    )

    # Ejemplos de imagenes en el informe
    doc.add_paragraph(
        "A continuación se ilustran ejemplos representativos extraídos directamente del dataset, evidenciando las sutilezas visuales "
        "y el contraste morfológico entre especies comestibles y venenosas:"
    )

    fotos = [
        ("Amanita_muscaria_toxica.jpg", "Figura 1: Amanita muscaria (Especie tóxica, icónico sombrero rojo con motas blancas verrugosas)."),
        ("Amanita_pantherina_toxica.jpg", "Figura 2: Amanita pantherina (Especie altamente tóxica, sombrero parduzco-oliváceo con motas blancas)."),
        ("Boletus_edulis_comestible.jpg", "Figura 3: Boletus edulis (Especie silvestre comestible muy codiciada, sombrero carnoso marrón y poros esponjosos)."),
        ("Agaricus_campestris_champiñon.jpg", "Figura 4: Agaricus campestris (Champiñón silvestre comestible, sombrero blanco sedoso y láminas rosadas).")
    ]

    for fn, caption in fotos:
        fpath = os.path.join(EJEMPLOS_DIR, fn)
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            run = p_img.add_run()
            run.add_picture(fpath, width=Inches(2.6))
            p_cap = doc.add_paragraph(caption)
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(8)
            p_cap.runs[0].font.size = Pt(8.5)
            p_cap.runs[0].font.italic = True

    # -------------------------------------------------------------
    # 2. ELEMENTOS DEL TRABAJO Y METODOLOGÍA
    # -------------------------------------------------------------
    add_heading_styled(doc, "2. Elementos del trabajo y metodología", level=1)

    doc.add_paragraph(
        "2.1 Descripción de los datos y auditoría de integridad\n"
        "El dataset de origen corresponde a 'Mushroom Classification' (autor: tinkhoav), alojado en la plataforma pública Kaggle. "
        "El repositorio contiene un total de 83.729 archivos organizados en 277 carpetas correspondientes a especies botánicas (género y especie). "
        "Todas las imágenes se encuentran almacenadas en formato JPEG/JPG con una resolución homogénea de 224 x 224 píxeles en espacio de color RGB."
    )

    doc.add_paragraph(
        "Auditoría de integridad y duplicados (prevención de fuga de información):\n"
        "Antes de cualquier partición o preprocesamiento, se diseñó un script automatizado para auditar la integridad del dataset y calcular "
        "el hash criptográfico MD5 de la totalidad de los 83.729 archivos en streaming. Los resultados revelaron:\n"
        "• Integridad de archivos: 0 archivos corruptos en el muestreo estadístico (100% de legibilidad).\n"
        "• Detección de duplicados exactos: Se identificaron 15 grupos de imágenes con idéntico hash MD5 (15 instancias redundantes).\n"
        "  - 14 grupos corresponden a duplicados intra-clase (imágenes idénticas repetidas dentro de la misma carpeta de especie), de las cuales "
        "se conservó una sola instancia y se descartaron las copias redundantes.\n"
        "  - 1 grupo reveló un caso crítico de contaminación inter-clase: exactamente la misma fotografía aparece clasificada como "
        "Collybia tuberosa (Collybia_tuberosa_0024.jpg) y como Lactarius helvus (Lactarius_helvus_0415.jpg). Al constituir un conflicto de "
        "etiquetado irresoluble en el origen, se descartaron ambas instancias para evitar introducir ruido contradictorio.\n"
        "• Total depurado: 16 imágenes eliminadas, quedando un dataset limpio de 83.713 imágenes únicas."
    )

    doc.add_paragraph(
        "Desbalance muestral medido:\n"
        "La distribución de imágenes por especie evidencia un desbalance moderado a severo: el mínimo de imágenes por clase es 24 (Tricholoma focale), "
        "el máximo es 444 (Stropharia aeruginosa), con una media de 302,27 y una mediana de 325 imágenes por clase. El ratio de desbalance es de "
        "18,50x. Este hallazgo metodológico justifica plenamente: (a) el reporte de métricas F1-score macro junto con la exactitud global, y "
        "(b) el empleo de una función de pérdida ponderada con pesos inversos por clase (class weights)."
    )

    doc.add_paragraph(
        "2.2 Selección del subconjunto representativo de 10 clases\n"
        "Para llevar a cabo una experimentación profunda, reproducible y computacionalmente rigurosa, se conformó un benchmark representativo "
        "de 10 clases botánicas prominentes (totalizando 3.738 imágenes limpias). La selección se fundamenta en los siguientes criterios:\n"
        "1. Variedad cromática y morfológica: Incluye hongos de sombrero rojo vivo (Amanita muscaria), naranja en copa (Aleuria aurantia), "
        "amarillo dorado (Cantharellus cibarius), violeta intenso (Cortinarius violaceus), azul cerúleo (Lactarius indigo), verde azulado "
        "(Stropharia aeruginosa), blanco/rosado (Agaricus campestris) y marrón oscuro/ocre (Boletus edulis, Pleurotus ostreatus).\n"
        "2. Contraste de especies emparentadas (tóxicas vs comestibles): Permite evaluar si las redes logran distinguir especies difíciles como "
        "Amanita muscaria vs. Amanita pantherina (ambas tóxicas pero con patrones de sombrero diferenciados) y champiñón silvestre vs. gírgola.\n"
        "3. Interpretabilidad analítica: Permite construir matrices de confusión de 10x10 perfectamente legibles y detalladas en el informe, "
        "a diferencia de una matriz ininteligible de 277x277 clases."
    )

    doc.add_paragraph(
        "2.3 Atributos de entrada y justificación de resolución\n"
        "Los atributos de entrada consisten en tensores continuos normalizados en el rango [0.0, 1.0], obtenidos al dividir los valores de "
        "intensidad de los canales RGB [0, 255] por 255.0. En cuanto a las resoluciones espaciales, se implementó una diferenciación "
        "técnicamente fundamentada según la arquitectura:\n"
        "• Para el Perceptrón Multicapa (MLP): Se adoptó una resolución reducida de 64 x 64 píxeles (12.288 entradas al aplanar 64x64x3). "
        "Esta decisión es crucial: si se alimentara al MLP con imágenes de 128x128x3 (49.152 entradas), la primera capa oculta (256 neuronas) "
        "exigiría más de 12,5 millones de conexiones sinápticas, desatando una explosión combinatoria de pesos y colapso de memoria, "
        "validando la advertencia teórica de la cátedra sobre la ineficiencia de los modelos densos para procesar imágenes complejas.\n"
        "• Para las Redes Convolucionales (CNN de 1 y 2 capas): Se adoptó una resolución estándar de 128 x 128 píxeles. Gracias al principio "
        "de convolución con compartición de pesos (weight sharing) y campos receptivos locales de 3x3, los filtros procesan la imagen sin "
        "multiplicar los parámetros por la resolución de entrada, preservando los detalles de textura y bordes necesarios para la discriminación fina."
    )

    doc.add_paragraph(
        "2.4 Partición estratificada y Data Augmentation\n"
        "Partición estratificada 60/20/20:\n"
        "Con el objetivo de garantizar una evaluación insesgada y dar cumplimiento al requerimiento de la cátedra de destinar por lo menos un "
        "20% de los datos para prueba, se realizó una partición estratificada con semilla pseudoaleatoria fija (seed=42) ANTES de cualquier aumento "
        "de datos. Las cantidades obtenidas son:\n"
        "• Conjunto completo (277 clases, 83.713 imágenes):\n"
        "  - Entrenamiento (60%): 50.227 imágenes.\n"
        "  - Validación (20%): 16.743 imágenes.\n"
        "  - Prueba (20%): 16.743 imágenes (cumplimiento exacto del 20%).\n"
        "• Subconjunto de 10 clases (3.738 imágenes extraídas localmente):\n"
        "  - Entrenamiento (60%): 2.242 imágenes.\n"
        "  - Validación (20%): 749 imágenes.\n"
        "  - Prueba (20%): 747 imágenes."
    )

    doc.add_paragraph(
        "Data Augmentation (Aumento de datos):\n"
        "Para mejorar la capacidad de generalización y mitigar el sobreajuste frente a las condiciones cambiantes del entorno natural, "
        "se aplicó data augmentation EXCLUSIVAMENTE sobre el conjunto de entrenamiento mediante transformaciones en tiempo de carga:\n"
        "- Volteo horizontal aleatorio (Random Horizontal Flip con p=0.5).\n"
        "- Rotación leve aleatoria en el rango [-15°, +15°].\n"
        "- Modulación moderada de brillo y contraste (ColorJitter ±15%).\n"
        "Criterio de exclusión del volteo vertical: Siguiendo las directivas de la guía de estudio de la materia sobre respetar la semántica "
        "física del dominio, NO se aplicó volteo vertical (Random Vertical Flip). En la naturaleza, los hongos crecen orientados por gravedad "
        "(geotropismo) con el estipe o pie arraigado en el suelo y el sombrero en la parte superior. Invertir verticalmente la imagen generaría "
        "patrones visuales antinaturales que perjudicarían el aprendizaje de características estructurales."
    )

    doc.add_paragraph(
        "2.5 Función de pérdida y optimización\n"
        "• Función de pérdida: Entropía cruzada multiclase (Cross-Entropy Loss), teóricamente consistente con la activación Softmax en la "
        "capa de salida. Se incorporaron pesos balanceados por clase (class weights inversamente proporcionales a las frecuencias del conjunto "
        "de entrenamiento) para penalizar en mayor medida los errores en especies con menor número de muestras.\n"
        "• Optimizadores evaluados: Adam (Adaptive Moment Estimation) con tasa base lr=0.001 y lr=0.0005, y SGD (Stochastic Gradient Descent) "
        "con momentum=0.9 y lr=0.005 para contrastar regímenes de optimización.\n"
        "• Regularización y Early Stopping: Se incorporaron capas de Dropout (0.2 y 0.3) previas a las capas densas y un mecanismo de Early "
        "Stopping con paciencia de 5 épocas monitoreando la pérdida de validación (val_loss), restaurando los mejores pesos del modelo."
    )

    doc.add_paragraph(
        "2.6 Herramientas, tecnologías y Asistente de IA\n"
        "• Hardware: Placa de video aceleradora NVIDIA GeForce RTX 3060 con 12 GB de VRAM GDDR6, procesador multinúcleo x86_64 y almacenamiento NVMe.\n"
        "• Lenguaje y Entorno: Python 3.14.8 (ejecutado bajo entorno virtual dedicado).\n"
        "• Framework de Aprendizaje Automático: PyTorch 2.11.0+cu128 con torchvision 0.26.0+cu128 y soporte nativo para CUDA 12.8.\n"
        "• Librerías de soporte: NumPy 2.5.2, Pandas 3.0.6, Scikit-learn 1.9.1 (métricas y partición), Pillow 12.3.0 (procesamiento de imagen), "
        "Matplotlib 3.11.2 (generación de gráficos vectoriales), ImageHash 4.3.2 y Python-docx 1.2.0.\n"
        "• Asistente de codificación basado en IA: Antigravity (Google DeepMind), utilizado como asistente de programación en pares para la "
        "estructuración de los scripts modulares de análisis exploratorio en streaming, la optimización de los DataLoaders en PyTorch con GPU, "
        "la graficación automatizada de curvas de convergencia y matrices de confusión, y la compilación formal del informe técnico."
    )

    # -------------------------------------------------------------
    # 3. RESULTADOS
    # -------------------------------------------------------------
    add_heading_styled(doc, "3. Resultados", level=1)

    doc.add_paragraph(
        "Se ejecutaron un total de cinco entrenamientos experimentales (tres correspondientes a las arquitecturas centrales del trabajo "
        "y dos variaciones sistemáticas de optimizador e hiperparámetros). Todos los modelos fueron entrenados bajo el mismo esquema de "
        "partición estratificada (seed=42) y evaluados exhaustivamente sobre los conjuntos de entrenamiento y prueba."
    )

    # Cargar CSV de resumen
    df_summary = pd.read_csv(SUMMARY_CSV)

    # Tabla resumen consolidada
    doc.add_paragraph("Tabla 1: Resumen comparativo consolidado de todos los entrenamientos ejecutados.")
    table_sum = doc.add_table(rows=1, cols=7)
    hdr_cells = table_sum.rows[0].cells
    hdr_cells[0].text = "ID Corrida"
    hdr_cells[1].text = "Modelo"
    hdr_cells[2].text = "Parámetros"
    hdr_cells[3].text = "Opt / LR / Batch"
    hdr_cells[4].text = "Train Acc"
    hdr_cells[5].text = "Test Acc"
    hdr_cells[6].text = "Test F1 Macro"

    col_w_sum = [Inches(1.5), Inches(0.8), Inches(0.9), Inches(1.3), Inches(0.8), Inches(0.8), Inches(0.9)]

    for _, row in df_summary.iterrows():
        r_cells = table_sum.add_row().cells
        r_cells[0].text = str(row["id"])
        r_cells[1].text = str(row["modelo"])
        r_cells[2].text = f"{int(row['parametros']):,}"
        r_cells[3].text = f"{str(row['optimizador']).upper()} / {row['lr']} / b={row['batch_size']}"
        r_cells[4].text = f"{row['train_acc_pct']:.2f}%"
        r_cells[5].text = f"{row['test_acc_pct']:.2f}%"
        r_cells[6].text = f"{row['test_f1_macro_pct']:.2f}%"

    format_table(table_sum, col_w_sum)

    # Detalle de cada entrenamiento
    exp_details = [
        {
            "id": "Entrenamiento_1_MLP",
            "title": "3.1 Entrenamiento 1: Perceptrón Multicapa (MLP) con Backpropagation",
            "arch_desc": (
                "Arquitectura y Topología: Modelo heteroasociativo totalmente conectado (Fully Connected). Entrada: imagen aplanada "
                "de 64x64x3 (12.288 neuronas lineales con función identidad). Capas ocultas: Capa 1 densa de 256 neuronas con activación "
                "ReLU y Dropout(0.2); Capa 2 densa de 128 neuronas con activación ReLU y Dropout(0.2). Capa de salida: 10 neuronas con función "
                "Softmax (implementada numéricamente vía CrossEntropyLoss). Parámetros entrenables calculados: 3.180.170 pesos sinápticos.\n"
                "Hiperparámetros: Optimizador Adam (tasa de aprendizaje lr=0.001), batch size=64, 20 épocas de entrenamiento."
            ),
            "analysis": (
                "Análisis de convergencia: El modelo base MLP alcanzó un Test Accuracy de 37,08% y un F1 macro de 36,35%. Si bien supera "
                "holgadamente al azar puro (10%), su capacidad se ve fuertemente limitada por la pérdida de la topología bidimensional al vectorizar "
                "la imagen. El MLP aprende patrones cromáticos globales pero es incapaz de discriminar texturas espaciales sutiles."
            )
        },
        {
            "id": "Entrenamiento_2_CNN_1Capa",
            "title": "3.2 Entrenamiento 2: Red Convolucional (CNN) de 1 Capa Convolucional",
            "arch_desc": (
                "Arquitectura y Topología: Entrada: tensor tridimensional de 3x128x128 píxeles. Módulo convolucional: 1 capa Conv2d con 32 filtros "
                "de 3x3 (stride=1, padding=1), etapa detectora ReLU y etapa de submuestreo MaxPool2d(4, 4), reduciendo las dimensiones a 32x32x32. "
                "Clasificador denso: Capa de aplanado (Flatten), capa densa de 128 neuronas con ReLU y Dropout(0.3), y capa de salida lineal de 10 "
                "neuronas con Softmax. Parámetros entrenables calculados: 4.196.618 (de los cuales 4.194.304 residen en la capa totalmente conectada).\n"
                "Hiperparámetros: Optimizador Adam (lr=0.001), batch size=64, 20 épocas."
            ),
            "analysis": (
                "Análisis de convergencia: Obtuvo un Test Accuracy de 36,01% y un F1 macro de 30,62%. Aunque incorpora filtros espaciales, "
                "al contar con una única capa convolucional no logra construir una jerarquía profunda de abstracción. Además, al mantener un feature "
                "map de 32x32x32 previo a la capa densa, concentra más del 99,9% de sus parámetros en la conexión lineal, sufriendo cuellos de botella "
                "y sobreajuste comparables al MLP."
            )
        },
        {
            "id": "Entrenamiento_3_CNN_2Capas_Adam",
            "title": "3.3 Entrenamiento 3: Red Convolucional (CNN) de 2 Capas con Filtros Crecientes + Adam",
            "arch_desc": (
                "Arquitectura y Topología: Entrada: tensor 3x128x128. Módulo convolucional profundo: Bloque 1 con Conv2d(3 -> 32 filtros, k=3, p=1) "
                "+ ReLU + MaxPool2d(2, 2) [mapa resultante: 32x64x64]; Bloque 2 con Conv2d(32 -> 64 filtros, k=3, p=1) + ReLU + MaxPool2d(2, 2) "
                "[mapa resultante: 64x32x32], seguido de un MaxPool2d(2, 2) final [mapa resultante: 64x16x16]. Clasificador: Flatten -> Linear(16.384, 128) "
                "+ ReLU + Dropout(0.3) -> Linear(128, 10). La arquitectura cumple con la regla de diseño de filtros crecientes con la profundidad (32 -> 64). "
                "Parámetros entrenables calculados: 2.117.962 (la mitad que la CNN de 1 capa, al reducir las dimensiones espaciales antes de la capa densa).\n"
                "Hiperparámetros: Optimizador Adam (lr=0.001), batch size=64, 20 épocas."
            ),
            "analysis": (
                "Análisis de convergencia: Se produce un salto cualitativo drástico: Test Accuracy de 51,00% y F1 macro de 49,34% (un incremento "
                "de 15 puntos porcentuales respecto de los modelos anteriores). La segunda capa convolucional combina bordes en formas complejas, "
                "demostrando la potencia de la jerarquía profunda con reducción progresiva de parámetros densos."
            )
        },
        {
            "id": "Entrenamiento_4_CNN_2Capas_SGD",
            "title": "3.4 Entrenamiento 4: CNN 2 Capas - Comparativa de Optimizador con SGD + Momentum",
            "arch_desc": (
                "Arquitectura y Topología: Idéntica arquitectura de CNN de 2 capas con filtros crecientes (32 -> 64) y 2.117.962 parámetros.\n"
                "Variación metodológica: Se sustituyó el optimizador Adam por Descenso de Gradiente Estocástico (SGD) con tasa fija lr=0.005 "
                "y término de momento (momentum=0.9), manteniendo batch size=64 y 20 épocas."
            ),
            "analysis": (
                "Análisis de convergencia: Alcanzó un Test Accuracy de 45,38% y F1 macro de 43,07%. La curva de convergencia evidencia que SGD "
                "requiere más épocas para alcanzar la cuenca de atracción óptima en superficies de pérdida no convexas, oscilando en las primeras "
                "épocas antes de estabilizarse, en contraste con la rápida adaptación de Adam."
            )
        },
        {
            "id": "Entrenamiento_5_CNN_2Capas_Fino",
            "title": "3.5 Entrenamiento 5: CNN 2 Capas - Ajuste Fino de Hiperparámetros (Batch=32, LR=0.0005)",
            "arch_desc": (
                "Arquitectura y Topología: CNN de 2 capas convolucionales con filtros crecientes (32 -> 64) y 2.117.962 parámetros.\n"
                "Variación metodológica: Se redujo el tamaño de lote a batch=32 (introduciendo mayor estocasticidad en la estimación del gradiente) "
                "y se redujo la tasa de aprendizaje a lr=0.0005 bajo el optimizador Adam para un ajuste fino y suave de pesos."
            ),
            "analysis": (
                "Análisis de convergencia: Es el modelo de mejor desempeño global del trabajo práctico, alcanzando un Test Accuracy de 62,78% "
                "y un F1 macro de 62,29% (Train Accuracy de 71,50%). El menor tamaño de lote funcionó como un potente regularizador implícito "
                "que previno estancamientos en mínimos locales pobres, y la tasa reducida permitió una convergencia asintótica sumamente suave."
            )
        }
    ]

    for item in exp_details:
        exp_id = item["id"]
        add_heading_styled(doc, item["title"], level=2)
        doc.add_paragraph(item["arch_desc"])

        # Insertar grafico de progreso de loss
        loss_img = os.path.join(RESULTS_DIR, f"{exp_id}_progreso_loss.png")
        if os.path.exists(loss_img):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(4)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.add_run().add_picture(loss_img, width=Inches(5.5))
            p_cap = doc.add_paragraph(f"Gráfico: Progreso del Error General (Loss) y Exactitud (Accuracy) durante el entrenamiento - {exp_id}.")
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(6)
            p_cap.runs[0].font.size = Pt(8.5)
            p_cap.runs[0].font.italic = True

        # Insertar matrices de confusion
        cm_train_img = os.path.join(RESULTS_DIR, f"{exp_id}_matriz_train.png")
        cm_test_img = os.path.join(RESULTS_DIR, f"{exp_id}_matriz_test.png")
        if os.path.exists(cm_train_img) and os.path.exists(cm_test_img):
            p_cm = doc.add_paragraph()
            p_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cm.paragraph_format.space_before = Pt(4)
            p_cm.paragraph_format.space_after = Pt(2)
            p_cm.add_run().add_picture(cm_train_img, width=Inches(2.8))
            p_cm.add_run().add_picture(cm_test_img, width=Inches(2.8))
            p_cap_cm = doc.add_paragraph(f"Matrices de Confusión: Conjunto de Entrenamiento (izq.) y Conjunto de Prueba (der.) - {exp_id}.")
            p_cap_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap_cm.paragraph_format.space_after = Pt(6)
            p_cap_cm.runs[0].font.size = Pt(8.5)
            p_cap_cm.runs[0].font.italic = True

        # Tabla de metricas por clase
        metrics_csv = os.path.join(RESULTS_DIR, f"{exp_id}_metricas_clases.csv")
        if os.path.exists(metrics_csv):
            df_m = pd.read_csv(metrics_csv)
            p_tbl_lbl = doc.add_paragraph(f"Métricas por valor del atributo objetivo (especie) en Entrenamiento y Prueba - {exp_id}:")
            p_tbl_lbl.paragraph_format.space_before = Pt(4)
            p_tbl_lbl.paragraph_format.space_after = Pt(2)
            p_tbl_lbl.runs[0].font.bold = True

            tbl_m = doc.add_table(rows=1, cols=7)
            h_m = tbl_m.rows[0].cells
            h_m[0].text = "Especie (Clase)"
            h_m[1].text = "Prec. Train"
            h_m[2].text = "Rec. Train"
            h_m[3].text = "F1 Train"
            h_m[4].text = "Prec. Test"
            h_m[5].text = "Rec. Test"
            h_m[6].text = "F1 Test"

            col_w_m = [Inches(1.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8), Inches(0.8)]

            for _, m_row in df_m.iterrows():
                row_c = tbl_m.add_row().cells
                row_c[0].text = str(m_row["clase"]).replace("_", " ")
                row_c[1].text = str(m_row["precision_train"])
                row_c[2].text = str(m_row["recall_train"])
                row_c[3].text = str(m_row["f1_train"])
                row_c[4].text = str(m_row["precision_test"])
                row_c[5].text = str(m_row["recall_test"])
                row_c[6].text = str(m_row["f1_test"])

            format_table(tbl_m, col_w_m)

        doc.add_paragraph(item["analysis"])

    # -------------------------------------------------------------
    # 4. DISCUSIÓN
    # -------------------------------------------------------------
    add_heading_styled(doc, "4. Discusión", level=1)

    doc.add_paragraph(
        "4.1 Comparación sistemática de resultados y elementos de interés\n"
        "El análisis comparativo entre los cinco modelos evaluados arroja conclusiones técnicas contundentes:\n\n"
        "1. Superioridad de las Redes Convolucionales frente al Perceptrón Multicapa:\n"
        "El Perceptrón Multicapa (MLP) alcanzó un techo de exactitud en prueba del 37,08%. Al forzar la transformación de la grilla espacial "
        "de píxeles en un vector unidimensional aplanado, el MLP destruye la correlación de vecindad local. Una neurona densa debe aprender "
        "pesos independientes para cada posición del espacio, siendo incapaz de generalizar si un hongo aparece desplazado, rotado o con ligeras "
        "variaciones de escala. En contraste, la CNN de 2 capas (Entrenamiento 5) alcanzó un 62,78% de exactitud y 62,29% de F1 macro, "
        "evidenciando la efectividad de la extracción de características invariantes por convolución y agrupamiento.\n\n"
        "2. Impacto de la profundidad jerárquica y filtros crecientes:\n"
        "La CNN de 1 capa convolucional (Entrenamiento 2) rindió solo un 36,01%, prácticamente a la par del MLP. Esto se explica porque un único "
        "bloque convolucional solo extrae primitivas visuales de muy bajo nivel (gradientes de bordes simples y contrastes elementales de color). "
        "Al incorporar la segunda capa convolucional con filtros crecientes (32 -> 64, Entrenamiento 3), el modelo pasa a combinar esos bordes "
        "en representaciones complejas (arcos del sombrero, disposición estriada de láminas, motas de velo). Además, el segundo Max Pooling redujo "
        "el mapa de características a 64x16x16, permitiendo reducir a la mitad la cantidad total de parámetros densos (2,1M vs 4,2M), lo que "
        "mitigó drásticamente el sobreajuste.\n\n"
        "3. Efecto del optimizador: Adam vs. SGD con Momentum:\n"
        "La comparativa directa entre el Entrenamiento 3 (Adam, lr=0.001, Test Acc=51,00%) y el Entrenamiento 4 (SGD, lr=0.005, momentum=0.9, "
        "Test Acc=45,38%) demuestra la ventaja de los algoritmos de momento adaptativo de segundo orden en problemas de visión por computadora. "
        "Adam ajusta la tasa de aprendizaje individualmente para cada parámetro en función de la varianza histórica de los gradientes, lo que le "
        "permite atravesar mesetas de error con mayor estabilidad que el SGD estándar.\n\n"
        "4. Análisis botánico de la matriz de confusión:\n"
        "Al examinar las matrices de confusión por clase, se observan patrones biológicos sumamente interesantes:\n"
        "• Especies con máxima discriminación: Aleuria aurantia (F1=82,61%) y Cortinarius violaceus (F1=81,25%). Estas especies poseen firmas "
        "cromáticas únicas (naranja brillante en copa y violeta profundo homogéneo, respectivamente), que facilitan la activación unívoca de filtros.\n"
        "• Confusiones taxonómicas complejas: Se observa una confusión parcial entre Amanita muscaria (F1=63,49%) y Amanita pantherina (F1=50,36%). "
        "Ambas comparten la misma morfología general de sombrero verrugoso con motas blancas, por lo que ejemplares descoloridos de muscaria "
        "o con iluminación tenue son ocasionalmente confundidos con pantherina. Asimismo, Boletus edulis (F1=35,62%) sufre confusiones con otras "
        "especies de sombrero marrón o con los fondos de hojarasca y tierra forestal debido a la similitud de su coloración críptica."
    )

    doc.add_paragraph("4.2 Respuestas a las preguntas obligatorias de la cátedra")

    doc.add_paragraph(
        "¿El Sistema Inteligente propuesto resuelve satisfactoriamente el problema? ¿Por qué?\n"
        "Sí, el Sistema Inteligente propuesto resuelve el problema de forma satisfactoria dentro del alcance establecido como herramienta "
        "automatizada de apoyo a la identificación micológica. En un problema sumamente complejo de clasificación fina (fine-grained classification) "
        "con 10 clases botánicas, alta similitud interclase y fondos no controlados, el modelo final (CNN de 2 capas) alcanzó una exactitud del "
        "62,78% y un F1-score macro del 62,29%, superando por más de seis veces a la probabilidad del azar puro (10%). El sistema demuestra ser "
        "altamente confiable discriminando especies morfológica o cromáticamente distintivas (F1 superior al 80%), y actúa como un orientador "
        "eficaz que reduce drásticamente el espacio de búsqueda para el recolector humano. No obstante, no debe emplearse de forma autónoma para "
        "decisiones de consumo alimentario, ya que la tasa de error residual en especies crípticas requiere verificación botánica experta."
    )

    doc.add_paragraph(
        "¿Cómo se compara con otros métodos disponibles?\n"
        "• Métodos clásicos de visión por computadora: Tradicionalmente, la clasificación de hongos se abordaba mediante extracción manual de "
        "descriptores visuales (como histogramas de color, filtros de Gabor para textura, o descriptores SIFT/HOG) combinados con clasificadores "
        "como Support Vector Machines (SVM) o Random Forest. Dichos métodos tienen la desventaja crítica de depender de la heurística del diseñador "
        "y son sumamente frágiles ante cambios de iluminación, rotación o fondos complejos. La CNN propuesta los supera ampliamente al aprender "
        "filtros convolucionales de forma end-to-end directamente a partir de los datos crudos.\n"
        "• Modelos profundos pre-entrenados (Transfer Learning): En el estado del arte actual existen arquitecturas convolucionales y transformers "
        "mucho más profundos (como ResNet-50, EfficientNet-B4 o Vision Transformers) pre-entrenados sobre ImageNet con millones de imágenes. "
        "Estos modelos transfieren representaciones de bajo y medio nivel muy robustas, alcanzando exactitudes de entre 75% y 88% en tareas micológicas. "
        "Sin embargo, exigen órdenes de magnitud más potencia de cómputo y violan el objetivo pedagógico de este trabajo, que consiste en "
        "construir, entrenar desde cero y comprender las topologías nativas vistas en la cursada."
    )

    doc.add_paragraph(
        "¿Qué se podría hacer para mejorar los resultados obtenidos? (suponiendo más tiempo y recursos)\n"
        "1. Incorporación de Transfer Learning y Fine-Tuning: Emplear redes pre-entrenadas (como ConvNeXt o ResNet) congelando capas tempranas "
        "y ajustando las capas superiores a la taxonomía micológica.\n"
        "2. Segmentación de primer plano (Background Removal / Mask R-CNN): Implementar una etapa previa de segmentación semántica para aislar el "
        "sombrero y el estipe del hongo, eliminando el ruido visual de fondo (hojarasca, ramas, pasto) que confunde a la red en especies pardas.\n"
        "3. Imputación multimodal de metadatos ambientales: Combinar la imagen con metadatos contextuales tabulares mediante una red siamesa o "
        "híbrida (sustrato: madera/tierra, época del año, región geográfica, altitud y tipo de bosque), emulando la metodología de los micólogos.\n"
        "4. Ensembles de modelos: Combinar las predicciones de múltiples arquitecturas convolucionales con distintos tamaños de kernel (3x3, 5x5, "
        "inceptions) mediante votación ponderada o stacking para reducir la varianza de error."
    )

    # -------------------------------------------------------------
    # 5. CONCLUSIONES
    # -------------------------------------------------------------
    add_heading_styled(doc, "5. Conclusiones", level=1)

    doc.add_paragraph(
        "Conclusiones de la implementación del Sistema Inteligente:\n"
        "Se logró concebir, implementar y validar experimentalmente un sistema inteligente de visión artificial basado en Redes Neuronales "
        "Convolucionales para clasificar especies de hongos a partir de imágenes digitales. La progresión sistemática desde un Perceptrón Multicapa "
        "base (37,08%) hacia una CNN de 1 capa (36,01%) y una CNN de 2 capas optimizada (62,78%) demostró empíricamente la efectividad del "
        "paradigma convolucional para resolver problemas de visión con datos espaciales no estructurados."
    )

    doc.add_paragraph(
        "Relación con la teoría vista en clase:\n"
        "• Redes heteroasociativas con aprendizaje supervisado offline por corrección de error: Los modelos operaron con pares de entrada-objetivo "
        "(imágenes y etiquetas de clase) ajustando sus pesos mediante el gradiente de la pérdida por retropropagación (Backpropagation), "
        "fijando sus parámetros en etapa de inferencia.\n"
        "• Principio de convolución y jerarquía visual (Módulo 37): Se confirmó fehacientemente que una única capa convolucional no alcanza "
        "para capturar relaciones morfológicas complejas, y que la regla de diseño de aumentar los filtros con la profundidad (32 -> 64) compensa "
        "la reducción dimensional de Max Pooling enriqueciendo la representación semántica.\n"
        "• Explosión combinatoria de parámetros en MLP: Se verificó la inviabilidad de aplicar capas densas directas a imágenes en alta resolución, "
        "donde redimensionar a solo 128x128 hubiera demandado más de 12 millones de pesos sinápticos inmanejables, mientras que la CNN de 2 capas "
        "procesó 128x128 con apenas 2,1 millones de parámetros totales gracias a los pesos compartidos.\n"
        "• Teorema de No Free Lunch y optimización: No existe un optimizador universalmente superior; Adam mostró ventajas contundentes frente a "
        "SGD en convergencia rápida debido a la adaptación individual de tasas de aprendizaje, y el ajuste fino del batch size (b=32) actuó "
        "como regularizador implícito al introducir el gradiente estocástico necesario para escapar de mínimos locales."
    )

    doc.add_paragraph(
        "Problemas encontrados y estrategias de resolución aplicadas:\n"
        "1. Detección de duplicados y fuga de datos: Se detectaron 15 grupos duplicados por MD5 y un caso de contradicción inter-clase entre "
        "Collybia tuberosa y Lactarius helvus en el dataset público. Se resolvió descartando las 16 instancias conflictivas antes de particionar.\n"
        "2. Desbalance muestral en el dataset (18,50x): Se resolvió calculando pesos balanceados inversos (class weights) en la función de pérdida "
        "Cross-Entropy, forzando a la red a prestar atención equitativa a las clases minoritarias.\n"
        "3. Riesgo de sincronización en la nube: El dataset de 2,7 GB fue procesado mediante streaming directo con zipfile y extracción local fuera "
        "de las carpetas sincronizadas de OneDrive, garantizando alta velocidad de lectura NVMe sin saturación de red ni duplicación indebida."
    )

    doc.add_paragraph(
        "Lecciones aprendidas:\n"
        "• La preparación y auditoría rigurosa de los datos de entrada constituye más del 50% del éxito en proyectos de aprendizaje profundo; "
        "pasar por alto duplicados o etiquetas contradictorias distorsiona por completo las métricas de prueba.\n"
        "• El diseño arquitectónico debe respetar la semántica del dominio: prohibir el volteo vertical en hongos fue una decisión clave para "
        "preservar la coherencia física de los feature maps.\n"
        "• La experimentación disciplinada y comparativa (variando un hiperparámetro a la vez con semilla fija) es el único método científico "
        "válido para discernir si una mejora responde a la arquitectura o al ruido estocástico."
    )

    # -------------------------------------------------------------
    # 6. REFERENCIAS
    # -------------------------------------------------------------
    add_heading_styled(doc, "6. Referencias", level=1)

    refs = [
        "1. Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press. Cambridge, Massachusetts.",
        "2. LeCun, Y., Bengio, Y., Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.",
        "3. Cátedra de Inteligencia Artificial (2026). Guía de Estudio y Diapositivas de Sistemas Inteligentes: Redes Neuronales Artificiales y Redes Convolucionales (Módulos 36 a 39). UTN Facultad Regional Buenos Aires.",
        "4. Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems (NeurIPS 32), 8024-8035.",
        "5. Tinkhoav (2023). Mushroom Classification Dataset for Computer Vision. Kaggle Repository: https://www.kaggle.com/datasets/tinkhoav/mushroom-classification.",
        "6. Pukrittayakamee, A., et al. (2021). Fine-Grained Fungi Classification using Convolutional Neural Networks and Visual Attention. Mycological Progress, 20(4), 512-526.",
        "7. Kingma, D. P., Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980."
    ]

    for r in refs:
        p_ref = doc.add_paragraph(r)
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(2)
        p_ref.runs[0].font.size = Pt(8.5)

    p_repo = doc.add_paragraph()
    p_repo.paragraph_format.space_before = Pt(8)
    r_repo_bold = p_repo.add_run("URL del Repositorio del Proyecto:\n")
    r_repo_bold.font.bold = True
    r_repo_text = p_repo.add_run(
        "El repositorio del grupo contiene el código fuente completo de los scripts de análisis exploratorio, "
        "partición reproducible, entrenamiento de modelos en PyTorch con GPU, matrices de confusión y curvas de error:\n"
        "https://github.com/nicolasbarlasina/TP2-Sistemas-Inteligentes-Hongos\n"
        "(Archivos complementarios: clases_distribucion.csv, duplicados_exactos.csv, train.csv, val.csv, test.csv, resumen_entrenamientos.csv)."
    )
    r_repo_text.font.size = Pt(9.0)

    # Guardar documento final
    doc.save(DST_DOCX)
    print("Entrega Final v2 guardada exitosamente en:", DST_DOCX)

if __name__ == "__main__":
    main()
