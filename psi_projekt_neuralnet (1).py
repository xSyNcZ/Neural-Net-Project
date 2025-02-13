# -*- coding: utf-8 -*-
"""PSI_projekt_neuralnet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F2-r0BYyncrZf4saM_kWt4RZvn2x33NH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_excel("/content/clean_data_water_quality_fin.xlsx")
df.head()

#df = pd.get_dummies(df, columns=["Year","Month","Day"], drop_first=True)

X = df.drop(columns=["Site_Id"])
y = df["Site_Id"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert text to integers
y_onehot = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train)

nnModel = Sequential([
    Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.35),
    Dense(128, activation='relu'),
    Dropout(0.35),
    Dense(64, activation='relu'),
    Dropout(0.35),
    Dense(y_onehot.shape[1], activation='softmax')
])

nnModel.compile(optimizer=Adam(learning_rate=0.0015), loss='categorical_crossentropy', metrics=['accuracy'])

nnModel.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2)

loss, accuracy = nnModel.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

y_pred = nnModel.predict(X_test_scaled)
y_pred_labels = label_encoder.inverse_transform(y_pred.argmax(axis=1))

y_test_labels = label_encoder.inverse_transform(y_test.argmax(axis=1))
for true_val, pred_val in zip(y_test_labels[:5], y_pred_labels[:5]):
    print(f"True Value: {true_val}, Predicted Value: {pred_val}")

# Przewidywanie etykiet dla zbioru testowego
# Model zwraca prawdopodobieństwa dla każdej klasy, dlatego wybieramy indeks z największym prawdopodobieństwem
y_pred_classes = y_pred.argmax(axis=1)  # Indeksy przewidywanych klas
y_test_classes = y_test.argmax(axis=1)  # Indeksy rzeczywistych klas

# Generowanie macierzy błędów
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Wyświetlanie macierzy błędów za pomocą funkcji ConfusionMatrixDisplay
# Dodanie etykiet klas dla lepszej czytelności (jeśli są dostępne w LabelEncoder)
class_labels = label_encoder.classes_  # Pobranie oryginalnych nazw klas z LabelEncoder

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='viridis', xticks_rotation='vertical')  # Wizualizacja z kolorowym gradientem
plt.title("Macierz Błędów dla Klasyfikacji Wieloklasowej")
plt.show()