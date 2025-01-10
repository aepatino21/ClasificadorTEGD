import pandas as pd
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import tensorflow as tf
from skimage import io
from skimage.transform import resize
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear un DataFrame con la tabla atributo-valor
data = {
    'COL': [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    'PCO': [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    'PFA': [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'ROC': [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
    'RCR': [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'RAN': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    'PAN': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'PLU': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'SOM': [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'PNA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'OPE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'PHO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ZAR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ABR': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'PUL': [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Etnias': ["Akawayo", "Kari'ña", "Arawak", "E'ñepa", "Mapoyo", "Yabarana", "Jivi", "Jodi", "Pemón", "Puinave", "Piaroa", "Warao", "Yanomami", "Ye'kwana-Sanoma"]
}

# Este código carga tu tabla atributo-valor en un DataFrame de pandas para facilitar su manejo y procesamiento.
df = pd.DataFrame(data)

# Cargado de datos desde los.csv
train_df = pd.read_csv('tribes_train.csv')
val_df = pd.read_csv('tribes_val.csv')
test_df = pd.read_csv('tribes_test.csv')


# Separar las etiquetas multiples
train_df['labels'] = train_df['labels'].apply(lambda x: x.split(','))
val_df['labels'] = val_df['labels'].apply(lambda x: x.split(','))
test_df['labels'] = test_df['labels'].apply(lambda x: x.split(','))

# Aumento de Datos mediante el generador de imagenes
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador para conjunto de entrenamiento
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='url',
    y_col='labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Generador para conjunto de validacion
val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col='url',
    y_col='labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
