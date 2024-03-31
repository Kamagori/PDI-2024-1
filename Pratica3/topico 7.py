import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Carrega dataset
dataset = pd.read_csv('dados.csv', header=None)

#  (B, G, R)
features = dataset.iloc[:, :-1]
labels = dataset.iloc[:, -1]

# Separa o dataset em dois grupos para o treinamento
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define o parametro de busca no "grid"
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (200,), (50, 50), (100, 100)],
    'learning_rate_init': [0.001, 0.01]
}

# Multi-Layer Perceptron (MLP) Classifier
mlp_classifier = MLPClassifier(max_iter=1000)

#grid search
grid_search = GridSearchCV(mlp_classifier, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

#Encontra parametro
best_params = grid_search.best_params_

# Aplica os melhores parametros a MLP
best_mlp_classifier = MLPClassifier(max_iter=1000, **best_params)
best_mlp_classifier.fit(X_train, y_train)

# Faz predicoes
mlp_predictions_train = best_mlp_classifier.predict(X_train)

# Calcula a predicao
mlp_train_precision = accuracy_score(y_train, mlp_predictions_train)

print("Best MLP Parameters:", best_params)
print("MLP Training Precision:", mlp_train_precision)

mlp_predictions_test = best_mlp_classifier.predict(X_test)

# Calcula a precisao do teste
mlp_test_precision = accuracy_score(y_test, mlp_predictions_test)

# Printa a matriz final de confusao
mlp_confusion_matrix = confusion_matrix(y_test, mlp_predictions_test)
print("MLP Confusion Matrix:")
print(mlp_confusion_matrix)
