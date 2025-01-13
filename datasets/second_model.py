import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np

# Leer y procesar los archivos .csv
train_csv_path = 'tribes_train.csv'
val_csv_path = 'tribes_val.csv'

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Transformar etiquetas en formato de one-hot encoding
train_labels = train_df['labels'].str.get_dummies(sep=',')
val_labels = val_df['labels'].str.get_dummies(sep=',')

train_df = train_df.join(train_labels)
val_df = val_df.join(val_labels)

# Verifica el número de etiquetas únicas
num_labels = train_labels.shape[1]

# Generar un flujo de datos a partir de los dataframes
# Definir el generador de datos con aumentación para el entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255
)

# Generador para el conjunto de validación con solo normalización
val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='url',
    y_col=train_labels.columns.tolist(),
    target_size=(224, 224),
    batch_size=16, # Probemos a reducir el tamano del batch a la mitad
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col='url',
    y_col=val_labels.columns.tolist(),
    target_size=(224, 224),
    batch_size=32, # Probemos a reducir el tamano del batch a la mitad
    class_mode='raw'
)

# Construir el modelo usando MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Añadir Dropout
predictions = Dense(num_labels, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implementar ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Entrenar el modelo
# Aumentemos el numero de epocas para ver si con mas entrenamiento el modelo prospera, si no lo hace, bajar el numero de epocas
model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[reduce_lr])

# Ultimo recurso para aumentar la precision desde la modificacion del modelo
# Descongelar algunas capas superiores del modelo base para ajuste fino
# for layer in base_model.layers[-20:]:      # Ajusta este valor según sea necesario
#   layer.trainable = True
#
# Recompilar el modelo con una tasa de aprendizaje más baja
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
# Continuar entrenando el modelo
# model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[reduce_lr])

# Evaluar el modelo
loss, accuracy = model.evaluate(val_generator)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

# Guardar el modelo
model.save('second_model.keras')
