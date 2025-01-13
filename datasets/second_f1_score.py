import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Cargar el conjunto de datos de prueba (asegúrate de que tiene el mismo formato que el conjunto de entrenamiento)
test_csv_path = 'tribes_test.csv'
test_df = pd.read_csv(test_csv_path)
test_labels = test_df['labels'].str.get_dummies(sep=',')
test_df = test_df.join(test_labels)

datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col='url',
    y_col=test_labels.columns.tolist(),
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    shuffle=False  # Importante para mantener el orden de las imágenes
)

# Cargar el modelo
model = load_model('second_modelv3.keras')

# Acumular predicciones y etiquetas verdaderas en lotes
all_true_classes = []
all_predicted_classes = []

for i in range(len(test_generator)):
    imgs, labels = next(test_generator)
    predictions = model.predict(imgs)

    # Convertir las predicciones y etiquetas verdaderas a clases
    all_true_classes.extend(labels.argmax(axis=1))
    all_predicted_classes.extend(predictions.argmax(axis=1))

# Calcular el F1 score
f1 = f1_score(all_true_classes, all_predicted_classes, average='weighted')
print(f'F1 Score: {f1}')
