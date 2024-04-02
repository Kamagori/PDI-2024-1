#pip install torch torchvision
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import datasets, transforms
import torch  # Importando o torch
import matplotlib.pyplot as plt
import seaborn as sns

# Transformações para pré-processar os dados
transform = transforms.Compose([
    transforms.ToTensor(),  # Converter as imagens para tensores
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Baixar o conjunto de dados CIFAR-10 e aplicar as transformações
trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
# Reduzindo o tamanho do conjunto de treinamento e teste
trainset_small, _ = torch.utils.data.random_split(trainset, [10000, len(trainset) - 10000])
testset_small, _ = torch.utils.data.random_split(testset, [2000, len(testset) - 2000])
# Carregar os dados em tensores 
trainloader = torch.utils.data.DataLoader(trainset_small, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset_small, batch_size=4, shuffle=False)

# Extrair os dados de treinamento e teste
X_train = []
y_train = []
for images, labels in trainloader:
    X_train.extend(images.numpy())
    y_train.extend(labels.numpy())
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = []
y_test = []
for images, labels in testloader:
    X_test.extend(images.numpy())
    y_test.extend(labels.numpy())
X_test = np.array(X_test)
y_test = np.array(y_test)

# Achatar as imagens
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Treinar o SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
# Fazer previsões
y_pred = svm.predict(X_test)

# Avaliação do modelo
print("Classification report:\n", classification_report(y_test, y_pred))
# Avaliar o desempenho
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


conf_matrix = confusion_matrix(y_test, y_pred)
# Plotando a matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão')
plt.show()