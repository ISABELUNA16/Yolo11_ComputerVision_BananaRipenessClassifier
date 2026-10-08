import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO

DETECTOR_PATH = "yolo11n.pt"
POSITIVES_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset_split/test")
NEGATIVES_DIR = Path("dataset_negatives")
TARGET_RECALL = 0.98     # queremos aceptar al menos el 98 % de los plátanos
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)


def max_banana_conf(model, banana_ids, paths, batch=32):
    """Corre el detector UNA vez con umbral bajo y guarda la mejor confianza por imagen."""
    scores = []
    for s in range(0, len(paths), batch):
        results = model.predict([str(p) for p in paths[s:s + batch]],
                                conf=0.01, classes=banana_ids, verbose=False)
        scores += [float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in results]
    return np.array(scores)


def main():
    model = YOLO(DETECTOR_PATH)
    banana_ids = [k for k, v in model.names.items() if v.lower() == "banana"]

    pos_paths, neg_paths = list_images(POSITIVES_DIR), list_images(NEGATIVES_DIR)
    if not pos_paths or not neg_paths:
        raise SystemExit(f"Faltan imágenes: {len(pos_paths)} positivas en {POSITIVES_DIR}, "
                         f"{len(neg_paths)} negativas en {NEGATIVES_DIR}")

    pos = max_banana_conf(model, banana_ids, pos_paths)
    neg = max_banana_conf(model, banana_ids, neg_paths)
    print(f"Positivos: {len(pos)} | Negativos: {len(neg)}\n")

    print(f"{'Umbral':>7} | {'Plátanos aceptados':>18} | {'No-plátanos aceptados':>21}")
    print("-" * 54)
    best = None
    for t in np.arange(0.05, 0.80, 0.05):
        tpr = (pos >= t).mean()   # plátanos que pasan el filtro
        fpr = (neg >= t).mean()   # no-plátanos que se cuelan
        print(f"{t:7.2f} | {tpr:18.2%} | {fpr:21.2%}")
        if tpr >= TARGET_RECALL:
            best = t   # el umbral más alto que aún acepta >= TARGET_RECALL

    if best is None:
        print(f"\nNingún umbral acepta el {TARGET_RECALL:.0%} de los plátanos: "
              "el detector COCO no reconoce bien estos plátanos.")
    else:
        print(f"\nUmbral sugerido: det_conf = {best:.2f}  (ponlo en BananaPipeline)")

    # Peores casos, para revisarlos a mano y comentarlos en el informe
    print("\nPlátanos con MENOR confianza:")
    for i in np.argsort(pos)[:5]:
        print(f"  {pos[i]:.2f}  {pos_paths[i]}")
    print("No-plátanos con MAYOR confianza:")
    for i in np.argsort(-neg)[:5]:
        print(f"  {neg[i]:.2f}  {neg_paths[i]}")


if __name__ == "__main__":
    main()