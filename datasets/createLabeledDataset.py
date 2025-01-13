import os
import pandas as pd

# Path del directorio principal
base_dir = './labeled_data'

# Inicializar el contador de id
id_counter = 1

# Lista para almacenar los datos del csv
csv_data = []

# Recorrer todos los directorios y archivos
for subdir, dirs, files in os.walk(base_dir):
    for file in files:
        # Obtener el nombre del directorio (label)
        labels = os.path.basename(subdir).split()

        # Obtener la extensión del archivo
        file_extension = os.path.splitext(file)[1]

        # Generar el nuevo nombre de archivo
        new_file_name = f"{id_counter}{file_extension}"

        # Obtener la ruta completa del archivo
        old_file_path = os.path.join(subdir, file)
        new_file_path = os.path.join(subdir, new_file_name)

        # Renombrar el archivo
        os.rename(old_file_path, new_file_path)

        # Convertir labels a una cadena separada por comas
        labels_str = ','.join(labels)

        # Agregar los datos al csv
        csv_data.append([id_counter, id_counter, new_file_path, labels_str])

        # Incrementar el contador de id
        id_counter += 1

# Crear un DataFrame de pandas con los datos
df = pd.DataFrame(csv_data, columns=['id', 'name', 'url', 'labels'])

# Definir el nombre del archivo .csv
csv_file_name = 'data.csv'

# Guardar el DataFrame en un archivo .csv
df.to_csv(csv_file_name, index=False)

print(f'Archivo .csv generado: {csv_file_name}')
