from ultralytics import YOLO
import torch

def main():
    # 1. Configurar dispositivo (GPU CUDA / CPU)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Entrenando en dispositivo: {device}")

    # 2. Cargar modelo base preentrenado de clasificación
    # Clasificacion con: 'yolo11n-cls.pt' o 'yolov8n-cls.pt'
    model = YOLO("yolo11n-cls.pt")

    # 3. Entrenar el modelo
    results = model.train(
        data="dataset_split",          # Ruta a la carpeta que contiene 'train' y 'val'
        epochs=50,                     # 50 épocas para convergencia sin sobreajustar
        imgsz=256,                     # Resolución nativa de BananaImageBD (256x256)
        batch=32,                      # Tamaño de batch
        device=device,
        project="BananaRipenessClasifier", # Nombre del proyecto donde se guardarán los resultados
        name="yolo_cls_experiment",    # Subcarpeta del experimento
        exist_ok=True,                 # Sobrescribe la carpeta en vez de crear yolo_cls_experiment2
        workers=4,
        optimizer="AdamW",             # Con optimizer='auto' Ultralytics ignora lr0
        lr0=0.001,                     # Learning rate inicial (ahora sí se aplica)
        seed=42,                       # Reproducibilidad
        pretrained=True,                # Usar pesos preentrenados de ImageNet
        hsv_h=0.0,                     # Desactiva la alteración de Tono (Hue). Evita que verde se vuelva amarillo.
        hsv_s=0.1,                     # Pequeñas variaciones de saturación
        hsv_v=0.2,                     # Simula variaciones de brillo/iluminación en la nave industrial
        degrees=180,                   # Permite rotación completa para orientaciones en la faja
        fliplr=0.5,                    # Espejado horizontal
        flipud=0.5,                    # Espejado vertical
    )
    
    # 4. Validacion  del modelo y obteneción métricas (Top-1 Accuracy)
    metrics = model.val()
    print(f"Top-1 Accuracy en validación: {metrics.top1:.4f}")

    # 5. Inferencia en una nueva imagen
    img_test_path = "banana_ripe.jpg"
    prediction = model(img_test_path)

    # Mostrar la clase predicha y su confianza
    probs = prediction[0].probs
    top1_index = probs.top1
    class_name = model.names[top1_index]
    confidence = probs.top1conf.item()

    print(f"Predicción: {class_name} ({confidence * 100:.2f}% de confianza)")

if __name__ == "__main__":
    main()