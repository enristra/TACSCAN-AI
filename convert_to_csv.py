import numpy as np
import medmnist
from medmnist import ChestMNIST, INFO

info = INFO['chestmnist']
DataClass = getattr(medmnist, info['python_class'])


## Estrazione di immagini ed etichette
train_dataset = DataClass(split='train', download=False)

# get images (flattened) and labels.
# train_dataset.imgs è un array numpy di immagini 28 x 28 (512 immagini nel training set)
# .reshape(len(train_dataset), -1) appiattisce ciascuna immagine 2D in un vettore 1D lungo 784 (28 x 28)
# /255.0 normalizza i valori pixel (da 0-255 a 0-1)
images = train_dataset.imgs.reshape(len(train_dataset), -1) / 255.0  # normalize
labels = train_dataset.labels.flatten()

# save to CSV
np.savetxt('X_train.csv', images, delimiter=',')
np.savetxt('y_train.csv', labels, fmt='%d')

print('Saved X_train.csv and y_train.csv!')
