import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def main():
    # --- Positivos: fotos reales que contienen plátanos ---
    positivos = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Banana"],
        max_samples=200,
        shuffle=True,
        seed=42,
        dataset_name="gate_positivos",
    )
    positivos.export(
        export_dir="fotos_openimages_platanos",
        dataset_type=fo.types.ImageDirectory,
    )
    print(f"Positivos exportados: {len(positivos)}")

    # --- Negativos difíciles: otras frutas y verduras ---
    # Se incluye "Banana" en la descarga SOLO para poder detectar y excluir
    # las fotos donde aparece un plátano junto a otras frutas.
    negativos = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Mango", "Lemon", "Orange", "Apple", "Pear", "Cucumber",
                 "Zucchini", "Bell pepper", "Pineapple", "Banana"],
        max_samples=300,
        shuffle=True,
        seed=42,
        dataset_name="gate_negativos",
    )
    sin_platano = negativos.match(~F("ground_truth.detections.label").contains("Banana"))
    sin_platano.export(
        export_dir="dataset_negatives/openimages_frutas",
        dataset_type=fo.types.ImageDirectory,
    )
    print(f"Negativos exportados (sin plátano): {len(sin_platano)}")


if __name__ == "__main__":
    main()