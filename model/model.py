import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import classification_report, multilabel_confusion_matrix

# Cargar los datos desde los archivos CSV
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# Función personalizada para la codificación de etiquetas múltiples
def preprocess_labels(labels):
    label_list = labels.split(',')
    encoded_labels = [1 if label in label_list else 0 for label in ['human', 'animal', 'object', 'terrain']]
    return encoded_labels

# Aplicar la función a las columnas de etiquetas
train_df['labels'] = train_df['labels'].apply(preprocess_labels)
val_df['labels'] = val_df['labels'].apply(preprocess_labels)
test_df['labels'] = test_df['labels'].apply(preprocess_labels)

# Configurar los generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generadores
train_generator = train_datagen.flow_from_dataframe(train_df, x_col='path', y_col='labels', target_size=(224, 224), class_mode='raw', batch_size=32)
val_generator = val_datagen.flow_from_dataframe(val_df, x_col='path', y_col='labels', target_size=(224, 224), class_mode='raw', batch_size=32)
test_generator = test_datagen.flow_from_dataframe(test_df, x_col='path', y_col='labels', target_size=(224, 224), class_mode='raw', batch_size=32, shuffle=False)

# Cargar el modelo base ResNet50 preentrenado en ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Añadir capas de clasificación
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='sigmoid')(x)  # 4 salidas para 4 posibles etiquetas

# Modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ENTRENAMIENTO DEL MODELO
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# PRUEBAS

# Evaluación en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

# Predicciones
test_preds = model.predict(test_generator)

# Convertir las predicciones en etiquetas binarias (0 o 1)
threshold = 0.5
test_labels_pred = (test_preds > threshold).astype(int)

# Etiquetas verdaderas
test_labels_true = np.array([preprocess_labels(labels) for labels in test_df['labels']])

# Matriz de confusión para cada etiqueta
conf_matrices = multilabel_confusion_matrix(test_labels_true, test_labels_pred)

# Mostrar la matriz de confusión y el reporte para cada etiqueta
for i, label in enumerate(['human', 'animal', 'object', 'terrain']):
    print(f"Confusion Matrix for {label}:")
    print(conf_matrices[i])
    print(f"Classification Report for {label}:")
    print(classification_report(test_labels_true[:, i], test_labels_pred[:, i]))
