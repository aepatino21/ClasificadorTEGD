import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd

# Paso 3: Cargar y preprocesar las imágenes
datagen = ImageDataGenerator(rescale=1./255)

# Paso 4: Leer y procesar los archivos .csv
train_csv_path = 'train.csv'
val_csv_path = 'val.csv'

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Transformar etiquetas en formato de one-hot encoding
train_labels = train_df['labels'].str.get_dummies(sep=',')
val_labels = val_df['labels'].str.get_dummies(sep=',')

train_df = train_df.join(train_labels)
val_df = val_df.join(val_labels)

# Paso 5: Generar un flujo de datos a partir de los dataframes
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='url',
    y_col=train_labels.columns.tolist(),
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col='url',
    y_col=val_labels.columns.tolist(),
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw'
)

# Paso 6: Construir el modelo usando MobileNet
base_model = MobileNet(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_labels.shape[1], activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Paso 7: Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Paso 8: Entrenar el modelo
model.fit(train_generator, validation_data=val_generator, epochs=10, steps_per_epoch=train_df.shape[0] // 32, validation_steps=val_df.shape[0] // 32)

# Paso 9: Evaluar el modelo
loss, accuracy = model.evaluate(val_generator)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

# Guardar el modelo
model.save('first_model.h5')

# Para cargar el modelo más adelante
# from tensorflow.keras.models import load_model
# modelo_cargado = load_model('ruta/a/tu/first_model.h5')
# Evaluar el modelo cargado para confirmar que se ha cargado correctamente
# loss, accuracy = modelo_cargado.evaluate(val_generator)
# print(f'Pérdida: {loss}, Precisión: {accuracy}')
