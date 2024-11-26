import pandas as pd

# Create empty dataframe
data_frame = []

# List of all of our .csv files (datasets)
datasets = ['datasets/FotosIndigenas.csv', 'datasets/FotosIndigenas2.csv', 'datasets/ImagenesIndigenasSalazar.csv']

# Read and store all of our datasets in the data_frame list
for csv_file in datasets:
    data_chunk = pd.read_csv(csv_file, encoding='UTF-8')
    data_frame.append(data_chunk)

# Combine all data from the .csv files
combined_data = pd.concat(data_frame, ignore_index=True)

# Asignar valores incrementales a la primera columna
combined_data.iloc[:, 0] = range(1, len(combined_data) + 1)

# Saved combined dataset
combined_data.to_csv('datasets/CombinedFotosIndigenas.csv', index=False)