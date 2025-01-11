import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el conjunto de datos de prueba (asegúrate de que tiene el mismo formato que el conjunto de entrenamiento)
test_csv_path = 'tribes_test.csv'
test_df = pd.read_csv(test_csv_path)
test_labels = test_df['labels'].str.get_dummies(sep=',')
test_df = test_df.join(test_labels)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
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
model = load_model('second_model.h5')

# Evaluar el modelo en el conjunto de pruebas
loss, accuracy = model.evaluate(test_generator)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

# Reiniciar el generador para asegurarse de que esté al principio
test_generator.reset()

# Realizar predicciones con el modelo
predictions = model.predict(test_generator)

# Definir el umbral de predicción
threshold = 0.5

# Obtener las etiquetas predichas según el umbral
predicted_labels = (predictions > threshold).astype(int)

# Mostrar algunas imágenes con sus etiquetas predichas
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i >= len(test_generator):
        break
    img, true_labels = next(test_generator)
    img = img[0]
    true_labels = true_labels[0]
    ax.imshow(img)
    ax.axis('off')

    pred_labels = predicted_labels[i]
    true_labels_str = ', '.join(test_labels.columns[true_labels == 1])
    pred_labels_str = ', '.join(test_labels.columns[pred_labels == 1])
    ax.set_title(f'True: {true_labels_str}\nPred: {pred_labels_str}', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Guardar la figura como un archivo de imagen
plt.savefig('second_predicciones_random.png')
print("La figura se guardó como 'second_predicciones_random.png'")
