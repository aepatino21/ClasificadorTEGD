import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# Función para descargar imágenes desde una URL
def download_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error al descargar la imagen {url}: {e}")
        return None

# Función para agregar texto a la imagen
def add_text_to_image(image, text, position, font_size=20):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, (0, 0, 0), font=font)

# Función para procesar las imágenes
def process_images(csv_path, output_folder, page_size=(800, 800), image_size=(128, 128), spacing=10):
    # Leer el CSV
    df = pd.read_csv(csv_path)

    # Formatear valores asociados a las paginas    
    page_number = 1
    x_offset = spacing
    y_offset = spacing
    
    # Crear una página en blanco
    page = Image.new('RGB', page_size, color='white')

    # Lista para almacenar datos procesados
    processed_data = []
    
    # Iterar sobre las filas del DataFrame
    for index, row in df.iterrows():
        url = row['Link']
        img = download_image(url)
        
        if img:
            # Redimensionar la imagen
            img = img.resize(image_size)
            
            # Verificar si la imagen cabe en la página actual
            if x_offset + image_size[0] + spacing > page_size[0]:
                x_offset = spacing
                y_offset += image_size[1] + spacing

            if y_offset + image_size[1] + spacing > page_size[1]:
                # Guardar la página actual y crear una nueva
                page.save(os.path.join(output_folder, f'page_{page_number}.jpg'))
                page_number += 1
                page = Image.new('RGB', page_size, color='white')
                x_offset = spacing
                y_offset = spacing
            
            # Pegar la imagen en la página
            page.paste(img, (x_offset, y_offset))
            x_offset += image_size[0] + spacing
            
            # Agregar datos procesados a la lista
            processed_data.append({'image_path': os.path.join(output_folder, f'image_{index}.jpg')})  # Ajusta esto a tus columnas de etiquetas (agregar luego): , 'label': row['label']
    
    # Guardar la última página si no está vacía
    if y_offset > spacing or x_offset > spacing:
        page.save(os.path.join(output_folder, f'page_{page_number}.jpg'))
    
    # Crear un DataFrame con los datos procesados
    processed_df = pd.DataFrame(processed_data)
    
    # Guardar el DataFrame procesado en un archivo CSV
    processed_df.to_csv(os.path.join(output_folder, 'processed_dataset.csv'), index=False)

# Rutas y configuración
csv_path = 'datasets/CombinedFotosIndigenas.csv'
output_folder = 'image_handling/processed_images'

# Procesar las imágenes
process_images(csv_path, output_folder)
