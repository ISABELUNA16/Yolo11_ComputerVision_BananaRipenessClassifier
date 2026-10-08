import os
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from banana_pipeline import BananaPipeline

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador de Madurez | Plátanos",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    /* Estructura general */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    /* Contenedor tipo tarjeta */
    .custom-card {
        background-color: #1E222A;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Titulares e insignias dentro de la tarjeta */
    .card-header {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #A0AAB0;
        margin-bottom: 5px;
    }
    .card-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Reducción de márgenes y tamaños */
    h1 { font-size: 28px !important; }
    h2 { font-size: 20px !important; }
    h3 { font-size: 16px !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RUTAS Y CONFIGURACIÓN DE CONSTANTES
# -----------------------------------------------------------------------------
MODEL_PATH = "runs/classify/BananaRipenessClasifier/yolo_cls_experiment/weights/best.pt"
DETECTOR_PATH = "yolo11n.pt" # detector COCO preentrenado (clase 'banana')


# Diccionario para mapear las clases a nombres descriptivos y colores
CLASS_INFO = {
    "Green": {"label": "🟢 Verde / Inmaduro", "color": "#28a745"},
    "Semi-ripe": {"label": "🟡 Semi-maduro", "color": "#ffc107"},
    "Ripe": {"label": "🟠 Maduro (Listo para consumo)", "color": "#fd7e14"},
    "Overripe": {"label": "🔴 Sobremaduro", "color": "#dc3545"}
}

@st.cache_resource
def load_pipeline(cls_path, det_path):
    #Carga detector + clasificador una sola vez y los mantiene en cache
    if not os.path.exists(cls_path):
        return None
    return BananaPipeline(det_weights=det_path, cls_weights=cls_path)

# -----------------------------------------------------------------------------
# BARRA LATERAL (SIDEBAR) - Arquitectura y Estado
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://em-content.zobj.net/source/microsoft-teams/337/banana_1f34c.png", width=65)
    st.title("Panel de Control")
    st.markdown("---")
    
    st.markdown("### ⚙️ Arquitectura del Sistema")
    st.info("""
    **Pipeline en 2 Etapas:**
    1. 🔍 **Detección:** `YOLO11n` (Detección)
    2. 🏷️ **Clasificación:** `YOLO11n-cls` (Clasificación)
    """)
    
    st.markdown("---")
    st.caption("🔍 **Ruta del Modelo:**")
    st.code(MODEL_PATH, language="bash")

# -----------------------------------------------------------------------------
# VISTA PRINCIPAL
# -----------------------------------------------------------------------------
st.title("🍌 Clasificador del Grado de Madurez del Plátano")
st.markdown("Carga una imagen para evaluar automáticamente el estado de conservación e inventario.")
st.markdown("---")

pipeline = load_pipeline(MODEL_PATH, DETECTOR_PATH)

if pipeline is None:
    st.error(f"⚠️ No se encontró el archivo del modelo en `{MODEL_PATH}`. Asegúrate de haber completado el entrenamiento primero.")
else:
    # Maquetación en 2 Columnas
    col_left, col_right = st.columns([1.1, 1], gap="large")
    
    with col_left:
        st.subheader("📸 Imagen de Entrada")
        uploaded_file = st.file_uploader(
            "Selecciona o arrastra una imagen...",
            type=["jpg", "jpeg", "png", "webp"]
        )
        
        if uploaded_file is not None:
            image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            
            with st.spinner("Procesando detección de plátanos en la imagen..."):
                results = pipeline.predict(image)
            
            if not results:
                st.image(image, caption="Imagen cargada", use_container_width=True)
                st.warning("⚠️ No se detectó ningún plátano en la imagen. Intenta con una foto más clara o cercana.")
            else:
                # Anotación de cajas delimitadoras
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                line_width = max(3, image.width // 200)
                
                for i, r in enumerate(results, 1):
                    color = CLASS_INFO.get(r["label"], {}).get("color", "#007bff")
                    draw.rectangle(r["box"], outline=color, width=line_width)
                    draw.text((r["box"][0] + 8, r["box"][1] + 8), f"#{i}", fill=color)
                
                st.image(annotated, caption=f"Detección completada: {len(results)} plátano(s) en escena", use_container_width=True)

    with col_right:
        st.subheader("📊 Diagnóstico y Resultados")
        
        if uploaded_file is None:
            st.info("👈 Por favor, carga una imagen en el panel izquierdo para desplegar los resultados.")
        elif results:
            # Selector de plátanos detectados si hay más de uno
            if len(results) > 1:
                selected_idx = st.selectbox(
                    "Selecciona el plátano a inspeccionar:",
                    options=list(range(len(results))),
                    format_func=lambda x: f"Plátano #{x+1} - ({CLASS_INFO.get(results[x]['label'], {}).get('label', results[x]['label'])})"
                )
            else:
                selected_idx = 0
                
            r = results[selected_idx]
            info = CLASS_INFO.get(r["label"], {"label": r["label"], "color": "#007bff"})
            
            # Vista previa del recorte individual
            st.image(r["crop"], caption=f"Recorte individual: Plátano #{selected_idx + 1}", width=180)
            
            # Tarjeta métrica de clasificación principal
            st.markdown(f"""
                <div class="custom-card" style="border-left: 6px solid {info['color']};">
                    <div class="card-header">Diagnóstico Principal</div>
                    <div class="card-value" style="color: {info['color']};">{info['label']}</div>
                    <div style="font-size: 14px; color: #E2E8F0;">
                        <b>Confianza del Clasificador:</b> {r['cls_conf'] * 100:.2f}%<br>
                        <span style="font-size: 12px; color: #A0AAB0;">Confianza de Detección: {r['det_conf'] * 100:.1f}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Distribución de probabilidades por categoría
            st.markdown("### Desglose de Probabilidades")
            for class_name_raw, conf in r["probs"].items():
                item_info = CLASS_INFO.get(class_name_raw, {"label": class_name_raw, "color": "#007bff"})
                val = float(conf)
                
                st.write(f"**{item_info['label']}** — `{val * 100:.1f}%`")
                st.progress(min(val, 1.0))