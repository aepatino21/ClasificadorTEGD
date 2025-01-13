import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Calcular la matriz de confusión
cm = confusion_matrix(all_true_classes, all_predicted_classes)

# Crear y guardar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_labels.columns)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Matriz de Confusión')
plt.savefig('matriz_confusion.png')
print("La matriz de confusión se guardó como 'matriz_confusion.png'")

