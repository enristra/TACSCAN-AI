import medmnist
from medmnist import ChestMNIST, INFO

info = INFO['chestmnist']
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=False)
val_dataset = DataClass(split='val', download=False)
test_dataset = DataClass(split='test', download=False)

print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")
print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
