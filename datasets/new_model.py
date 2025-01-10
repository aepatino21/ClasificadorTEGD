import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

# Lectura de los archivos .csv
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')


def prepare_data(df, all_labels, label_dict):
    df['labels'] = df['labels'].apply(lambda x: x.split(','))

    def one_hot_encode(labels):
        encoded = np.zeros(len(all_labels), dtype=np.float32)
        for label in labels:
            if label in label_dict:
                encoded[label_dict[label]] = 1.0
            else:
                print(f"Etiqueta desconocida: {label}")
        return encoded
    df['one_hot_labels'] = df['labels'].apply(one_hot_encode)
    df['one_hot_labels'] = df['one_hot_labels'].apply(lambda x: np.array(x)) # Convertir a lista
    return df

# Obtener todas las etiquetas y el diccionario de etiquetas
all_labels = set(label for sublist in train_data['labels'].apply(lambda x: x.split(',')) for label in sublist)
label_dict = {label: idx for idx, label in enumerate(all_labels)}

# Preparar datos de entrenamiento y validación
train_data = prepare_data(train_data, all_labels, label_dict)
val_data = prepare_data(val_data, all_labels, label_dict)
#print(train_data.head())
#print(train_data.tail())
#print(val_data.head())
#print(val_data.tail())

#print(label_dict)

# Verificar tipos de datos
#print(train_data['one_hot_labels'].apply(lambda x: isinstance(x, np.ndarray)).all()) # Si son
#print(val_data['one_hot_labels'].apply(lambda x: isinstance(x, np.ndarray)).all()) # Si son


# Creacion del Generador de Imagenes
datagen = ImageDataGenerator(rescale=1./255)

# Generador para los datos de entrenamiento
train_generator = datagen.flow_from_dataframe(
      train_data,
      x_col='url',
      y_col = 'one_hot_labels'.tolist(),
      target_size=(224,224),
      batch_size=32,
      class_mode='raw'  # raw para multietiquetas
)

# Generador para los datos de validacion
val_generator = datagen.flow_from_dataframe(
      val_data,
      x_col='url',
      y_col='one_hot_labels'.tolist(),
      target_size=(224,224),
      batch_size=32,
      class_mode='raw' # raw para multietiquetas
)

# Cargar el modelo ResNet50 preentrenado
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas del modelo base para aprovechar los pesos preentrenados
for layer in base_model.layers:
    layer.trainable = False

# Construir el modelo (Agregar capas adicionales a las de ResNet50)
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(all_labels), activation='sigmoid') # sigmoid para salidas multietiqueta
])

# Compilacion del modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(
      train_generator,
      epochs=20,    # El modelo se entrenara en 20 epocas
      validation_data=val_generator
)

# Guardemos el modelo
model.save('first_model.h5')

# Suponiendo que tienes una nueva imagen procesada
# # predicciones = first_model.predict(nueva_imagen)
