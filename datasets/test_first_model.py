import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paso 1: Cargar el modelo guardado
model = load_model('first_modelv2.h5')

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

# Paso 5: Evaluar el modelo en el conjunto de pruebas
loss, accuracy = model.evaluate(test_generator)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

# Reiniciar el generador para asegurarse de que esté al principio
test_generator.reset()

# Acumular todas las imágenes y etiquetas verdaderas del generador
all_images = []
all_true_labels = []

for _ in range(len(test_generator)):
    imgs, labels = next(test_generator)
    all_images.append(imgs)
    all_true_labels.append(labels)

all_images = np.vstack(all_images)
all_true_labels = np.vstack(all_true_labels)

# Seleccionar 9 índices aleatorios
random_indices = np.random.choice(len(all_images), 9, replace=False)

# Crear una figura grande para contener todas las sub-imágenes
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axs.flat):
    img_idx = random_indices[i]
    ax.imshow(all_images[img_idx])

    # Obtener la predicción
    prediction = model.predict(np.expand_dims(all_images[img_idx], axis=0))[0]
    predicted_label = test_labels.columns[prediction.argmax()]
    true_label = test_labels.columns[all_true_labels[img_idx].argmax()]

    ax.set_title(f'Pred: {predicted_label}, True: {true_label}')
    ax.axis('off')

# Guardar la figura como un archivo de imagen
plt.savefig('predicciones_random.png')
print("La figura se guardó como 'predicciones_random.png'")

