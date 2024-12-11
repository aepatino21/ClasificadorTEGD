import struct
import numpy as np
import matplotlib.pyplot as plt

def read_rgb_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Leer la cabecera mágica, número de imágenes, filas y columnas (ajustado para RGB)
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))

        # Leer el resto del archivo para obtener los píxeles de las imágenes
        image_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape a (num_images, num_rows, num_cols//3, 3)
        images = image_data.reshape((num_images, num_rows, num_cols // 3, 3))

        return images

# Ruta al archivo idx3-ubyte
file_path = 'mnist_data.idx3-ubyte'

# Leer las imágenes
images = read_rgb_idx3_ubyte(file_path)

# Mostrar la primera imagen como ejemplo
plt.imshow(images[2])
plt.title("Primera Imagen del Dataset (128x128 RGB)")
plt.show()
