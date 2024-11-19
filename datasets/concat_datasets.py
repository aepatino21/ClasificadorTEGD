import pandas as pd

# Create empty dataframe
data_frame = []

# List of all of our .csv files (datasets)
datasets = ['datasets/FotosIndigenas.csv', 'datasets/FotosIndigenas2.csv']

# Read and store all of our datasets in the data_frame list
for csv_file in datasets:
    data_chunk = pd.read_csv(csv_file)
    data_frame.append(data_chunk)

# Combine all data from the .csv files
combine_data = pd.concat(data_frame, ignore_index=True)

# Saved combined dataset
combine_data.to_csv('datasets/CombinedFotosIndigenas.csv', index=False)