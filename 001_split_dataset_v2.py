import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

# (70% Train, 15% Val, 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Classes
CLASSES = ["Green", "Semi-ripe", "Ripe", "Overripe"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Todas las copias con el mismo prefijo son el MISMO plátano original,
# por lo que deben ir juntas al mismo split (si no, podría haber de datos).
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


def original_id(path: Path) -> str:
    """'Green_100_jpg.rf.0aad...' -> 'Green_100_jpg'"""
    return RF_SUFFIX.sub("", path.stem)


def split_dataset(source_dir: str, output_dir: str, seed: int = 42):

    rng = random.Random(seed)  # semilla aleatoria = reproducibilidad

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Borrar el split anterior (y sus train.cache / val.cache) para no mezclar
    if output_path.exists():
        shutil.rmtree(output_path)

    splits = ["train", "val", "test"]

    # Crear carpetas destino para cada split y clase
    for split in splits:
        for cls in CLASSES:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    print("Iniciando la reorganización del dataset (agrupada por plátano original)...\n")

    assigned = {split: set() for split in splits}  # para verificar fuga al final
    total_copied = 0

    for cls in CLASSES:
        cls_folder = source_path / cls

        if not cls_folder.exists():
            print(f"Error: No se encontró la carpeta '{cls}' en {source_path}")
            continue

        # Agrupar las imágenes por plátano original
        groups = defaultdict(list)
        for img in sorted(cls_folder.iterdir()):
            if img.suffix.lower() in VALID_EXTENSIONS:
                groups[original_id(img)].append(img)

        # Mezclar los IDs originales (no los archivos)
        ids = sorted(groups)
        rng.shuffle(ids)

        n = len(ids)
        train_count = int(n * TRAIN_RATIO)
        val_count = int(n * VAL_RATIO)

        parts = {
            "train": ids[:train_count],
            "val": ids[train_count:train_count + val_count],
            "test": ids[train_count + val_count:],
        }

        n_images = sum(len(v) for v in groups.values())
        print(f"Clase '{cls}': {n} plátanos originales ({n_images} imágenes)")

        for split, split_ids in parts.items():
            copied = 0
            for oid in split_ids:
                for img in groups[oid]:
                    shutil.copy(img, output_path / split / cls / img.name)
                    copied += 1
                assigned[split].add(f"{cls}/{oid}")
            print(f"  ├── {split.capitalize():<5}: {len(split_ids):4d} originales -> {copied:4d} imágenes")

        total_copied += n_images

    # Verificación: ningún plátano original puede estar en dos splits
    overlap = (
        (assigned["train"] & assigned["val"])
        | (assigned["train"] & assigned["test"])
        | (assigned["val"] & assigned["test"])
    )
    if overlap:
        raise RuntimeError(f"Fuga de datos detectada: {sorted(overlap)[:5]}")

    print(f"\nProceso completado. Se organizaron {total_copied} imágenes en '{output_dir}'.")
    print("Verificación OK: ningún plátano original aparece en más de un split.")


if __name__ == "__main__":

    SOURCE_DIRECTORY = "dataset_origin"
    OUTPUT_DIRECTORY = "dataset_split"
    split_dataset(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)