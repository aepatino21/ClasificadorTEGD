import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd

# Paso 4: Leer y procesar los archivos .csv
train_csv_path = 'tribes_train.csv'
val_csv_path = 'tribes_val.csv'

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Transformar etiquetas en formato de one-hot encoding
train_labels = train_df['labels'].str.get_dummies(sep=',')
val_labels = val_df['labels'].str.get_dummies(sep=',')

train_df = train_df.join(train_labels)
val_df = val_df.join(val_labels)


# Verificar si tenemos 16 etiquetas únicas
#print(f'Unique train labels: {train_labels.columns.tolist()} (Count: {train_labels.shape[1]})')
#print(f'Unique val labels: {val_labels.columns.tolist()} (Count: {val_labels.shape[1]})')

# Asegurarse de tener exactamente 16 etiquetas
#assert train_labels.shape[1] == 12, "El número de etiquetas en el conjunto de entrenamiento no es 16"
#assert val_labels.shape[1] == 12, "El número de etiquetas en el conjunto de validación no es 16"

# Verifica el número de etiquetas únicas
num_labels = train_labels.shape[1]
# print(f'Número de etiquetas: {num_labels}')

# Paso 5: Generar un flujo de datos a partir de los dataframes
# Definir el generador de datos con aumentación
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

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
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_labels, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Paso 7: Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Paso 8: Entrenar el modelo
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Paso 9: Evaluar el modelo
loss, accuracy = model.evaluate(val_generator)
print(f'Pérdida: {loss}, Precisión: {accuracy}')

# Guardar el modelo
model.save('second_model.keras')
