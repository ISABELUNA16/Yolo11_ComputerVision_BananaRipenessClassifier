import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Paths

MODEL_PATH = "runs/classify/BananaRipenessClasifier/yolo_cls_experiment/weights/best.pt"
TEST_DIR = "dataset_split/test"
CONFUSION_MATRIX_SAVE_PATH = "confusion_matrix_test_v3.png"

def evaluate_without_sklearn():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"No se encontró el conjunto de prueba en: {TEST_DIR}")

    print("Cargando el modelo YOLO entrenado...")
    model = YOLO(MODEL_PATH)

    # Clases ordenadas
    class_names = [model.names[i] for i in sorted(model.names.keys())]
    num_classes = len(class_names)
    print(f"Clases a evaluar: {class_names}")

    # Matriz de confusión inicializada en cero [num_classes x num_classes]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    total_images = 0
    correct_predictions = 0
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    print("\nEjecutando inferencias en el conjunto de prueba...")
    for true_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(TEST_DIR, class_name)
        if not os.path.exists(class_folder):
            continue

        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(glob.glob(os.path.join(class_folder, ext)))

        print(f"Procesando {len(image_paths)} imágenes para '{class_name}'...")

        for img_path in image_paths:
            try:
                results = model(img_path, verbose=False)
                pred_idx = results[0].probs.top1

                # Incrementar conteo en la matriz de confusión (Fila: Real, Columna: Predicho)
                cm[true_idx, pred_idx] += 1
                total_images += 1
                if true_idx == pred_idx:
                    correct_predictions += 1
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")

    if total_images == 0:
        print("No se encontraron imágenes en la carpeta de prueba.")
        return

    # Accuracy Global
    accuracy = (correct_predictions / total_images) * 100
    print("\n" + "="*50)
    print(f" ACCURACY GLOBAL EN TEST: {accuracy:.2f}% ({correct_predictions}/{total_images})")
    print("="*50 + "\n")

    # Mapeo textual de métricas por clase
    print(f"{'Clase':<15} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 40)
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"{name:<15} | {precision*100:6.2f}%    | {recall*100:6.2f}%")

    # Guardar Matriz de Confusión gráfica con Matplotlib puro
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión - Test Set')
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Añadir valores dentro de los cuadros
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_SAVE_PATH, dpi=300)
    plt.close()

    print(f"\nMatriz de confusión: {CONFUSION_MATRIX_SAVE_PATH}")

if __name__ == "__main__":
    evaluate_without_sklearn()