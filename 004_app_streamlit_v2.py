import os
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from banana_pipeline import BananaPipeline

# Pagina principal

st.set_page_config(
    page_title="Clasificador de Madurez del Plátano",
    page_icon="🍌",
    layout="centered"
)

# Rutas de los modelos
MODEL_PATH = "runs/classify/BananaRipenessClasifier/yolo_cls_experiment/weights/best.pt"
DETECTOR_PATH = "yolo11n.pt"   # detector COCO preentrenado (clase 'banana')

# Diccionario para mapear las clases a nombres descriptivos y colores
CLASS_INFO = {
    "Green": {"label": "Verde / Inmaduro", "color": "#28a745"},
    "Semi-ripe": {"label": "Semi-maduro", "color": "#ffc107"},
    "Ripe": {"label": "Maduro (Listo para consumo)", "color": "#fd7e14"},
    "Overripe": {"label": "Sobremaduro", "color": "#dc3545"}
}


@st.cache_resource
def load_pipeline(cls_path, det_path):
    """Carga detector + clasificador una sola vez y los mantiene en caché."""
    if not os.path.exists(cls_path):
        return None
    return BananaPipeline(det_weights=det_path, cls_weights=cls_path)


# Interfaz de usuario

st.title("🍌 Clasificador del Grado de Madurez del Plátano")
st.write("Carga una imagen de un plátano para predecir su estado de madurez.")
st.write("Etapa 1: detección de plátanos · Etapa 2: clasificación de madurez (**YOLO11n** Ultralytics).")

pipeline = load_pipeline(MODEL_PATH, DETECTOR_PATH)

if pipeline is None:
    st.error(f"No se encontró el archivo del modelo en `{MODEL_PATH}`. Asegúrate de haber completado el entrenamiento primero.")
else:
    uploaded_file = st.file_uploader(
        "Selecciona una imagen...",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        # exif_transpose: corrige la rotación de las fotos de celular
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")

        with st.spinner("Buscando plátanos en la imagen..."):
            results = pipeline.predict(image)

        if not results:
            # Etapa 1 rechaza la imagen: no se clasifica nada
            st.image(image, caption="Imagen cargada", use_container_width=True)
            st.warning("No se detectó ningún plátano en la imagen. "
                       "Sube una foto donde el plátano se vea con claridad.")
        else:
            # Dibujar las cajas detectadas sobre la imagen original
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            line_width = max(2, image.width // 250)
            for i, r in enumerate(results, 1):
                color = CLASS_INFO.get(r["label"], {}).get("color", "#007bff")
                draw.rectangle(r["box"], outline=color, width=line_width)
                draw.text((r["box"][0] + 5, r["box"][1] + 5), f"#{i}", fill=color)

            st.image(annotated, caption=f"{len(results)} plátano(s) detectado(s)", use_container_width=True)

            # Diagnóstico de cada plátano detectado
            for i, r in enumerate(results, 1):
                info = CLASS_INFO.get(r["label"], {"label": r["label"], "color": "#007bff"})

                st.divider()
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.image(r["crop"], caption=f"Plátano #{i}", use_container_width=True)
                with col2:
                    st.write("### Diagnóstico")
                    st.markdown(
                        f"<h3 style='color: {info['color']};'>{info['label']}</h3>",
                        unsafe_allow_html=True
                    )
                    st.write(f"**Confianza:** {r['cls_conf'] * 100:.2f}%")
                    st.caption(f"Confianza de detección: {r['det_conf'] * 100:.1f}%")

                st.write("**Probabilidades por categoría:**")
                for class_name_raw, conf in r["probs"].items():
                    label = CLASS_INFO.get(class_name_raw, {}).get("label", class_name_raw)
                    st.write(f"**{label}** ({conf * 100:.1f}%)")
                    st.progress(min(float(conf), 1.0))