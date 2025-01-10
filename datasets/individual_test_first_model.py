import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Cargar el modelo guardado
model = load_model('first_modelv2.h5')

# Paso 2: Leer el archivo .csv de pruebas para obtener las etiquetas
test_csv_path = 'test.csv'
test_df = pd.read_csv(test_csv_path)

# Transformar etiquetas en formato de one-hot encoding (si es necesario)
test_labels = test_df['labels'].str.get_dummies(sep=',')
class_labels = test_labels.columns.tolist()

# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(image_path):
    # Cargar la imagen
    image = load_img(image_path, target_size=(224, 224))
    # Convertir la imagen a un array
    image_array = img_to_array(image)
    # Reescalar los valores de los píxeles
    image_array = image_array / 255.0
    # Expandir las dimensiones para que coincidan con las del modelo
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Ruta de la imagen que deseas probar
image_path = '../images/WhatsApp Image 2025-01-10 at 1.52.51 PM.jpeg'

# Cargar y preprocesar la imagen
preprocessed_image = load_and_preprocess_image(image_path)

# Realizar la predicción
prediction = model.predict(preprocessed_image)[0]

# Obtener la etiqueta predicha
predicted_label = class_labels[prediction.argmax()]

# Mostrar la imagen y la etiqueta predicha
image = load_img(image_path)
plt.imshow(image)
plt.title(f'Predicción: {predicted_label}')
plt.axis('off')

# Guardar la figura como un archivo de imagen
plt.savefig('predicciones_individual.png')
print("La figura se guardó como 'predicciones_individual.png'")
