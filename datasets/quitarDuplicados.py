import pandas as pd

# Se carga el archivo CombinedFotosIndigenas.csv
df = pd.read_csv(r'C:\Users\ceviche\Documents\ClasificadorTEGD\datasets\CombinedFotosIndigenas.csv')

# Elimina duplicados basados en el link
df_sin_duplicados = df.drop_duplicates(subset=['Link'])

# Guardar el DataFrame sin duplicados en un nuevo archivo CSV
df_sin_duplicados.to_csv('CombinedFotosIndigenas_sin_duplicados.csv', index=False)

print("Datos duplicados eliminados y guardados en 'CombinedFotosIndigenas_sin_duplicados.csv'") 