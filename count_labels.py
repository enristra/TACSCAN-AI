import numpy as np

# Carica etichette
labels = np.loadtxt('y_train.csv', dtype=int)

# Conta quante etichette 0 e quante 1
count_0 = np.sum(labels == 0)
count_1 = np.sum(labels == 1)

print(f"Numero etichette 0 (negativi): {count_0}")
print(f"Numero etichette 1 (positivi): {count_1}")
print(f"Totale: {len(labels)}")
