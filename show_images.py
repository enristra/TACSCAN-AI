import medmnist
from medmnist import ChestMNIST, INFO
import numpy as np
import matplotlib.pyplot as plt
import os

# Parametri
output_dir = 'exported_images'
num_images = 50  # Quante immagini salvare

# Crea cartella di output
os.makedirs(output_dir, exist_ok=True)

# Carica dataset
info = INFO['chestmnist']
DataClass = getattr(medmnist, info['python_class'])
dataset = DataClass(split='train', download=False)

# Salva immagini
for i in range(num_images):
    img, label = dataset[i]
    img = np.squeeze(img)  # da (1, 28, 28) a (28, 28)

    filename = f'image_{i:03d}_label_{label[0]}.png'
    filepath = os.path.join(output_dir, filename)

    plt.imsave(filepath, img, cmap='gray')

print(f'Salvate {num_images} immagini in \"{output_dir}\".')

