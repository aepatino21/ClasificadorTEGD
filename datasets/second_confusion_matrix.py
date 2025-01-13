import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cargar el conjunto de datos de prueba
test_csv_path = 'tribes_test.csv'
test_df = pd.read_csv(test_csv_path)
test_labels = test_df['labels'].str.get_dummies(sep=',')
test_df = test_df.join(test_labels)

# Preprocesamiento de imágenes
datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col='url',
    y_col=test_labels.columns.tolist(),
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    shuffle=False  # Mantener el orden de las imágenes
)

# Cargar el modelo
model = load_model('second_modelv69_definitive.keras')

# Acumular todas las imágenes y etiquetas verdaderas del generador
all_images = []
all_true_labels = []

for _ in range(len(test_generator)):
    imgs, labels = next(test_generator)
    all_images.append(imgs)
    all_true_labels.append(labels)

all_images = np.vstack(all_images)
all_true_labels = np.vstack(all_true_labels)

# Realizar predicciones con el modelo
predictions = model.predict(all_images)

# Definir el umbral de predicción
threshold = 0.5

# Obtener las etiquetas predichas según el umbral
predicted_labels = (predictions > threshold).astype(int)

# Convertir las etiquetas verdaderas y predichas a índices
true_indices = np.argmax(all_true_labels, axis=1)
pred_indices = np.argmax(predicted_labels, axis=1)

# Generar la matriz de confusión
conf_matrix = confusion_matrix(true_indices, pred_indices)

# Mostrar la matriz de confusión usando Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_labels.columns, yticklabels=test_labels.columns)
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.title('Matriz de Confusión')
plt.show()

# Guardar la figura como un archivo de imagen
plt.savefig('matriz_de_confusion.png')
print("La matriz de confusión se guardó como 'matriz_de_confusion.png'")
