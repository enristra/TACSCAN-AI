import numpy as np
import matplotlib.pyplot as plt

# Carica immagini e etichette
X = np.loadtxt('X_train.csv', delimiter=',')
y = np.loadtxt('y_train.csv', dtype=int)

# Controllo dimensioni
if X.shape[0] != y.shape[0]:
    print(f"Errore: numero righe diverso! X: {X.shape[0]}, y: {y.shape[0]}")
    exit()
else:
    print(f"OK: {X.shape[0]} immagini e {y.shape[0]} etichette allineate.")

# Estrai la prima immagine
idx = 0
img_flat = X[idx]
label = y[idx]

# Ricostruisci in 28x28
img_2d = img_flat.reshape(28, 28)

# Salva immagine su disco
filename = f'sample_{idx}_label_{label}.png'
plt.imsave(filename, img_2d, cmap='gray')

print(f"Salvata immagine {filename}")
