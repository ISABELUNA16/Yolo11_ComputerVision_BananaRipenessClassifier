import torch

# 1. Comprobar si CUDA está disponible
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))

# 2. Asignar el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
