import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import struct

# Ruta del archivo CSV que contiene las URLs
csv_file_path = '../datasets/CombinedFotosIndigenas_sin_duplicados.csv'

# Carpeta donde se almacenarán las imágenes descargadas
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)

# Leer URLs del archivo CSV
df = pd.read_csv(csv_file_path)
urls = df['Link'].tolist()

# Inicializar listas para datos de imágenes
images = []

# Descargar y preprocesar imágenes
for url in urls:
    url = url.strip()
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('L')  # Convertir a escala de grises
        img = img.resize((28, 28))  # Redimensionar a 28x28 píxeles
        images.append(np.array(img))
    except Exception as e:
        print(f"Error al descargar o procesar {url}: {e}")

# Convertir imágenes a un array numpy
images_array = np.array(images)

# Guardar datos en formato MNIST
def save_mnist_images(filename, images):
    with open(filename, 'wb') as f:
        # Cabecera mágica (2051)
        f.write(struct.pack('>IIII', 2051, len(images), 28, 28))
        # Imágenes en sí
        for img in images:
            f.write(img.tobytes())

output_file = 'mnist_data.idx3-ubyte'
save_mnist_images(output_file, images_array)

print(f"Imágenes guardadas en formato MNIST en {output_file}")
