# Clasificación Automática del Grado de Madurez del Plátano

Este proyecto implementa una solución completa de **Visión por Computadora** para la clasificación automática del grado de madurez del plátano (*Green*, *Semi-ripe*, *Ripe*, *Overripe*) utilizando un modelo de Deep Learning basado en **Ultralytics YOLO11n** y desplegado en una interfaz gráfica interactiva con **Streamlit**.

---

## 📌 Tabla de Contenidos
- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Dataset Utilizado](#-dataset-utilizado)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos e Instalación](#-requisitos-e-instalación)
- [Flujo de Trabajo del Proyecto](#-flujo-de-trabajo-del-proyecto)
  - [1. Preparación del Dataset](#1-preparación-del-dataset)
  - [2. Entrenamiento del Modelo](#2-entrenamiento-del-modelo)
  - [3. Evaluación del Modelo](#3-evaluación-del-modelo)
  - [4. Interfaz Gráfica con Streamlit](#4-interfaz-gráfica-con-streamlit)
- [Resolución de Problemas Frecuentes](#-resolución-de-problemas-frecuentes)
- [Compartir la Aplicación](#-compartir-la-aplicación)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)

---

## 📖 Descripción del Proyecto

Determinar con precisión el estado de madurez de las frutas es esencial para optimizar la cadena de suministro agrícola, la gestión de inventario y el control de calidad en centros de distribución. Este proyecto aplica técnicas de **Transfer Learning** sobre arquitecturas de clasificación de YOLO para predecir con alta confianza el grado de madurez de un plátano a partir de imágenes individuales.

---

## 📊 Dataset Utilizado

El proyecto utiliza el conjunto de datos **BananaImageBD (v2)** alojado en Mendeley Data:
- **Nombre:** `Augmented Banana Ripeness Detection Dataset.zip`
- **Formato:** Imágenes JPEG recortadas a $256 \times 256$ píxeles.
- **Clases (4):**
  - 🟢 **Green:** Inmaduro / Verde
  - 🟡 **Semi-ripe:** Semi-maduro
  - 🟠 **Ripe:** Maduro (Listo para consumo)
  - 🔴 **Overripe:** Sobremaduro

---

## 📂 Estructura del Proyecto

```text
BananaRipenessClassifier/
│
├── 01_split_dataset.py       # Script para dividir las imágenes en train/val/test
├── 02_yolo_train.py          # Script de entrenamiento con Ultralytics YOLO
├── 03_evaluate_model.py      # Script de evaluación de métricas y matriz de confusión
├── 04_app_streamlit.py       # Aplicación web interactiva con Streamlit
│
├── banana_dataset_split/     # Directorio generado con la estructura train/val/test
│   ├── train/
│   ├── val/
│   └── test/
│
├── banana_ripeness/          # Resultados y pesos guardados del entrenamiento
│   └── yolo_cls_experiment/
│       └── weights/
│           └── best.pt       # Pesos entrenados del modelo
│
├── confusion_matrix_test.png # Matriz de confusión generada en la evaluación
├── requirements.txt          # Dependencias del entorno
└── README.md                 # Documentación del proyecto
```

---

## 💻 Requisitos e Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/ISABELUNA16/Yolo11_ComputerVision_BananaRipenessClassifier.git
cd BananaRipenessClassifier
```

### 2. Crear y activar un entorno virtual
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
     
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias
```bash
pip install -r requirements.txt
```

#### Contenido de `requirements.txt`:
```text
ultralytics
streamlit
matplotlib
numpy
pillow
```

---

## 🚀 Flujo de Trabajo del Proyecto

### 1. Preparación del Dataset
Descarga el archivo [Augmented Banana Ripeness Detection Dataset.zip](https://data.mendeley.com/datasets/ptfscwtnyz/2) de Mendeley, descomprímelo en una carpeta local y ejecuta el script de división:


```bash
python 01_split_dataset.py
```
*Este script dividirá estratificadamente los datos en 70% entrenamiento (`train`), 15% validación (`val`) y 15% prueba (`test`).*

---

### 2. Entrenamiento del Modelo
Ejecuta el entrenamiento utilizando Transfer Learning con YOLO:

```bash
python 02_yolo_train.py
```
*El script ajustará las imágenes a $256 \times 256$ durante 30 épocas y guardará los mejores pesos en `banana_ripeness/yolo_cls_experiment/weights/best.pt`.*

> **Nota:** La exportación a ONNX es opcional. Si se omiten las dependencias nativas de exportación, los pesos en formato PyTorch (`best.pt`) son plenamente funcionales para evaluación y despliegue.

---

### 3. Evaluación del Modelo
Evalúa el rendimiento del modelo sobre el conjunto de pruebas independientes (`test`):

```bash
python 03_evaluate_model.py
```
**Resultados esperados:**
- Imprime en consola la precisión global ($Accuracy$) y métricas por clase ($Precision$ y $Recall$).
- Guarda en disco la **Matriz de Confusión** gráfica (`confusion_matrix_test.png`).

---

### 4. Interfaz Gráfica con Streamlit
Para lanzar el panel web interactivo e inferir sobre nuevas imágenes:

```bash
streamlit run 04_app_streamlit.py
```
*(Asegúrate de incluir el nombre exacto del archivo `.py` que contiene la interfaz).*

La aplicación estará disponible localmente en `http://localhost:8501`. Permite arrastrar cualquier imagen de un plátano, visualizar la clase predicha con su indicador de confianza y consultar la distribución de probabilidades.

---

## 🛠️ Resolución de Problemas Frecuentes

- **Directivas de seguridad en Windows (`ImportError: DLL load failed` / WDAC):**  
  Ocurre cuando las directivas de seguridad o antivirus bloquean librerías nativas o binarias comprimidas (`.pyd` / `.dll`). Se soluciona utilizando la implementación de evaluación ligera integrada en `03_evaluate_model.py` (basada en NumPy/Matplotlib puro) y reejecutando en entornos con permisos estándar.
- **Advertencias de Streamlit (`use_column_width` redefinido):**  
  En las versiones recientes de Streamlit, se debe reemplazar el parámetro `use_column_width=True` por `use_container_width=True` o `width="stretch"` en las llamadas a `st.image()`.
- **Archivo no encontrado al ejecutar Streamlit (`File does not exist`):**  
  Asegúrate de especificar el nombre exacto del script Python (por ejemplo, `04_app_streamlit.py`) al ejecutar `streamlit run`.

---

## 🧰 Tecnologías Utilizadas

- **Lenguaje:** Python 3.14
- **Deep Learning Framework:** Ultralytics YOLO
- **Procesamiento de Imágenes:** Pillow (PIL)
- **Cálculo Numérico & Gráficos:** NumPy, Matplotlib
- **Web App / UI:** Streamlit