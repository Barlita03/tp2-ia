# Informe de Análisis del Dataset: Mushroom Classification

## 1. Resumen General del Dataset

- **Origen:** Kaggle (`tinkhoav/mushroom-classification`).
- **Archivo comprimido:** `dataset.zip` (2.51 GB).
- **Cantidad total de archivos en el zip:** 83,729
- **Cantidad de imágenes clasificadas:** 83,729
- **Cantidad total de clases (especies):** 277
- **Metadatos y licencias en el zip:** No se encontraron archivos de texto (`README`, `LICENSE`). El repositorio de Kaggle indica dominio público / CC0 según autor, pero en el zip los metadatos de atribución individual no están incluidos.

## 2. Distribución y Balance de Clases

- **Mínimo de imágenes por clase:** 24
- **Máximo de imágenes por clase:** 444
- **Promedio por clase:** 302.27
- **Mediana por clase:** 325.0
- **Ratio de desbalance (Máx/Mín):** 18.50x
- *Nota metodológica:* Existe un desbalance moderado a severo entre clases (24 vs 444), lo cual justifica fuertemente reportar métricas macro (F1 macro) y considerar ponderación de clases (class weights) en el entrenamiento.

## 3. Integridad, Formatos y Modos de Color

- **Extensiones declaradas:** {'jpg': 42240, 'jpeg': 41252, 'png': 114, '<sin_extension>': 122, 'gif': 1}
- **Archivos sin extensión:** 122 (verificados: son archivos JPEG válidos a los que les falta la extensión en el nombre).
- **Archivos corruptos en muestra (N=2000):** 0 (0.0%).
- **Modos de color en muestra:** {'RGB': 2000}. La gran mayoría es `RGB`. Alrededor de un porcentaje menor requiere normalización explícita a 3 canales RGB (descartando alfa en RGBA o convirtiendo escala de grises L a RGB).
- **Rango de resoluciones:** Ancho entre 224 y 224 px; Alto entre 224 y 224 px. Mediana: 224x224 px.

## 4. Detección de Duplicados (Fuga de Información)

- **Imágenes únicas por hash MD5:** 83,714
- **Instancias duplicadas exactas:** 15
- **Duplicados intra-clase:** 14 grupos.
- **Duplicados inter-clase (conflicto de etiqueta):** 1 grupos.
- *Decisión de preprocesamiento:* Se deben eliminar los duplicados exactos **antes** de realizar la partición estratificada 60/20/20 para evitar data leakage entre conjuntos de entrenamiento y prueba.

## 5. Ejemplos Seleccionados para el Informe

Se seleccionaron y extrajeron 4 especies representativas en `ejemplos_informe/`:
1. `Amanita_muscaria_toxica.jpg`: hongo icónico de sombrero rojo con motas blancas, tóxico.
2. `Amanita_pantherina_toxica.jpg`: especie cercana y altamente tóxica, sombrero parduzco.
3. `Boletus_edulis_comestible.jpg`: hongo silvestre comestible de alta demanda gastronómica.
4. `Agaricus_campestris_champiñon.jpg`: champiñón silvestre comestible.
