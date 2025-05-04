import medmnist
from medmnist import ChestMNIST, INFO

info = INFO['chestmnist']
DataClass = getattr(medmnist, info['python_class'])

# download the dataset
train_dataset = DataClass(split='train', download=True)
test_dataset = DataClass(split='test', download=True)

print('Downloaded ChestMNIST!')
