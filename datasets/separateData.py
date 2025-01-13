import pandas as pd
import numpy as np

# Cargar el archivo CSV
df = pd.read_csv('tribes_data.csv')

# Separar aleatoriamente el DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Calcular los tamaños de los splits
train_size = int(0.70 * len(df))
val_size = int(0.15 * len(df))
test_size = len(df) - train_size - val_size

# Dividir el DataFrame en tres partes
train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

# Guardar cada parte en un nuevo archivo CSV
train_df.to_csv('tribes_train.csv', index=False)
val_df.to_csv('tribes_val.csv', index=False)
test_df.to_csv('tribes_test.csv', index=False)
