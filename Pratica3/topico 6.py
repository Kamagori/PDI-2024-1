import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Carrega o dataset
dataset = pd.read_csv('dados.csv', header=None)  # Presume o header

# (B, G, R)
features = dataset.iloc[:, :-1]  
labels = dataset.iloc[:, -1]  

# Divide o dataset pra ser treinado
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Vetor de duporte da SVM
svm_classifier = SVC(kernel='linear')  
svm_classifier.fit(X_train, y_train)  # Classificador dda SVM
svm_predictions = svm_classifier.predict(X_test)  

# Multi-Layer Perceptron (MLP) Classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000)  # Parametros que podem ser ajustados
mlp_classifier.fit(X_train, y_train)  # Treino de MLP 
mlp_predictions = mlp_classifier.predict(X_test)

# Calcula acuracia
svm_accuracy = accuracy_score(y_test, svm_predictions)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)

print("SVM Accuracy:", svm_accuracy)
print("MLP Accuracy:", mlp_accuracy)

# Printa matriz de confusao
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
mlp_confusion_matrix = confusion_matrix(y_test, mlp_predictions)

print("SVM Confusion Matrix:")
print(svm_confusion_matrix)
print("\nMLP Confusion Matrix:")
print(mlp_confusion_matrix)
