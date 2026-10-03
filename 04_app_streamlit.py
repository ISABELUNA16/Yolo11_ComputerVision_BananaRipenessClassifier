import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Pagina principal

st.set_page_config(
    page_title="Clasificador de Madurez del Plátano",
    page_icon="🍌",
    layout="centered"
)

# Ruta del modelo entrenado
MODEL_PATH = "runs/classify/BananaRipenessClasifier/yolo_cls_experiment/weights/best.pt"

# Diccionario para mapear las clases a nombres descriptivos y colores
CLASS_INFO = {
    "Green": {"label": "Verde / Inmaduro", "color": "#28a745"},
    "Semi-ripe": {"label": "Semi-maduro", "color": "#ffc107"},
    "Ripe": {"label": "Maduro (Listo para consumo)", "color": "#fd7e14"},
    "Overripe": {"label": "Sobremaduro", "color": "#dc3545"}
}

@st.cache_resource
def load_yolo_model(model_path):
    """Carga el modelo YOLO y lo mantiene en caché para mayor velocidad."""
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

# Interfez de usuario

st.title("🍌 Clasificador del Grado de Madurez del Plátano")
st.write("Carga una imagen de un plátano para predecir su estado de madurez.")
st.write("Modelo aplicado: **YOLO11n** Ultralytics.")

# Cargar modelo
model = load_yolo_model(MODEL_PATH)

if model is None:
    st.error(f" No se encontró el archivo del modelo en `{MODEL_PATH}`. Asegúrate de haber completado el entrenamiento primero.")
else:
    # Cargar imagen de prueba
    uploaded_file = st.file_uploader(
        "Selecciona una imagen...", 
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Imagen cargada", use_container_width=True)
        with col2:
            st.write("### Diagnóstico")
            with st.spinner("Procesando imagen ..."):
                # Inferencia con el modelo YOLO
                results = model(image, verbose=False)
                probs = results[0].probs

                # Obtener la clase predicha y la confianza
                top1_idx = probs.top1
                top1_conf = probs.top1conf.item()
                predicted_class_raw = model.names[top1_idx]

                # Mapear información de la clase
                info = CLASS_INFO.get(predicted_class_raw, {"label": predicted_class_raw, "color": "#007bff"})

                # Visualizar resultado principal
                st.markdown(
                    f"<h3 style='color: {info['color']};'>{info['label']}</h3>", 
                    unsafe_allow_html=True
                )
                st.write(f"**Confianza:** {top1_conf * 100:.2f}%")

        st.divider()
        st.write("### Probabilidades por categoría: ")

        # Mostrar barras de progreso para todas las clases
        for idx, class_name_raw in model.names.items():
            conf = probs.data[idx].item()
            label = CLASS_INFO.get(class_name_raw, {}).get("label", class_name_raw)
            
            st.write(f"**{label}** ({conf * 100:.1f}%)")
            st.progress(min(float(conf), 1.0))