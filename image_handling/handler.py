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
urls = df['Link'].tolist()  # Asegúrate de que la columna se llame 'url'

# Inicializar listas para datos de imágenes
images = []

# Descargar y preprocesar imágenes
for i, url in enumerate(urls):
    url = url.strip()
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))  # Mantener la imagen en color
        img = img.resize((128, 128))  # Redimensionar a 128x128 píxeles
        images.append(np.array(img))
    except Exception as e:
        print(f"Error al descargar o procesar {url}: {e}")

# Convertir imágenes a un array numpy
images_array = np.array(images)

# Guardar datos en formato MNIST modificado para imágenes RGB
def save_rgb_mnist_images(filename, images):
    with open(filename, 'wb') as f:
        # Cabecera mágica modificada para RGB (2051 -> imágenes en color)
        f.write(struct.pack('>IIII', 2051, len(images), 128, 128 * 3))  # Multiplica el tamaño de las columnas por 3 (RGB)
        # Imágenes en sí
        for img in images:
            f.write(img.tobytes())

output_file = 'mnist_data.idx3-ubyte'
save_rgb_mnist_images(output_file, images_array)

print(f"Imágenes guardadas en formato MNIST en {output_file}")
