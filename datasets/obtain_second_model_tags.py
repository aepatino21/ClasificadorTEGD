import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model('second_model.h5')

# Preprocesar la imagen
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Ruta de la imagen que deseas predecir
image_path = 'path/to/your/image.jpg'
image = preprocess_image(image_path)

# Hacer predicciones
predictions = model.predict(image)

# Obtener etiquetas con una probabilidad mayor a un umbral (ej. 0.5)
threshold = 0.5
predicted_labels = [train_labels.columns[i] for i, prob in enumerate(predictions[0]) if prob > threshold]

print(f'Etiquetas predichas: {predicted_labels}')
