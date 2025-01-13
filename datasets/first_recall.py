import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score

# Paso 1: Cargar el modelo guardado
model = load_model('first_modelv4.h5')

# Paso 2: Configurar ImageDataGenerator para el conjunto de pruebas
test_datagen = ImageDataGenerator(rescale=1./255)

# Paso 3: Leer el archivo .csv de pruebas
test_csv_path = 'test.csv'
test_df = pd.read_csv(test_csv_path)

# Transformar etiquetas en formato de one-hot encoding (si es necesario)
test_labels = test_df['labels'].str.get_dummies(sep=',')
test_df = test_df.join(test_labels)

# Paso 4: Generar un flujo de datos a partir del dataframe de pruebas
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col='url',
    y_col=test_labels.columns.tolist(),
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

# Acumular predicciones y etiquetas verdaderas en lotes
all_true_classes = []
all_predicted_classes = []

for i in range(len(test_generator)):
    imgs, labels = next(test_generator)
    predictions = model.predict(imgs)

    # Convertir las predicciones y etiquetas verdaderas a clases
    all_true_classes.extend(labels.argmax(axis=1))
    all_predicted_classes.extend(predictions.argmax(axis=1))

# Calcular el recall
recall = recall_score(all_true_classes, all_predicted_classes, average='weighted')
print(f'Recall: {recall}')
