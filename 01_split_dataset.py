import os
import shutil
import random
from pathlib import Path

# (70% Train, 15% Val, 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Classes
CLASSES = ["Green", "Semi-ripe", "Ripe", "Overripe"]

def split_dataset(source_dir: str, output_dir: str, seed: int = 42):
    
    random.seed(seed) # semilla aleatoria = reproducibilidad
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    splits = ["train", "val", "test"]
    
    # Crear carpetas destino para cada split y clase
    for split in splits:
        for cls in CLASSES:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
            
    print("Iniciando la reorganización del dataset...\n")
    
    total_copied = 0
    
    for cls in CLASSES:
        cls_folder = source_path / cls
        
        if not cls_folder.exists():
            print(f"Error: No se encontró la carpeta '{cls}' en {source_path}")
            continue
            
        # Obtener todas las imágenes compatibles
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG', '.JPEG')
        images = [f for f in os.listdir(cls_folder) if f.endswith(valid_extensions)]
        
        # Mezclar aleatoriamente las imágenes para evitar sesgos
        random.shuffle(images)
        
        total_images = len(images)
        train_count = int(total_images * TRAIN_RATIO)
        val_count = int(total_images * VAL_RATIO)
        
        # Asignar particiones
        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:]
        
        # Copiar imágenes a las carpetas correspondientes
        for img in train_imgs:
            shutil.copy(cls_folder / img, output_path / "train" / cls / img)
            
        for img in val_imgs:
            shutil.copy(cls_folder / img, output_path / "val" / cls / img)
            
        for img in test_imgs:
            shutil.copy(cls_folder / img, output_path / "test" / cls / img)
            
        print(f"Clase '{cls}': {total_images} imágenes en total")
        print(f"  ├── Train: {len(train_imgs)}")
        print(f"  ├── Val:   {len(val_imgs)}")
        print(f"  └── Test:  {len(test_imgs)}")
        
        total_copied += total_images

    print(f"\n Proceso completado exitosamente. Se organizaron {total_copied} imágenes en '{output_dir}'.")

if __name__ == "__main__":

    SOURCE_DIRECTORY = "dataset_origin"    
    OUTPUT_DIRECTORY = "dataset_split"
    split_dataset(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)