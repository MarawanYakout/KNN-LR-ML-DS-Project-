import os
import pandas as pd
# FILE CHECKER FOR INTEGRATY 

file_list = [r'organized_data\gdp_rate_worldwide.csv',r'organized_data\migration_oecd.csv']
missing_files = []

for file in file_list:
    if not os.path.exists(file):
        missing_files.append(file)
        for file_missing in missing_files:
            print(f'File {file_missing} is missing or not available')
    else:
        print(f"Files Loaded Sucessfully")

loaded_data = {}
for file in file_list:
    if os.path.exists(file):
        loaded_data[file] = pd.read_csv(file)
        print(f"Loaded data from {file}")
    else:
        print(f"File {file} is missing or not available, could not load data.")

# Print top 10 values in loaded data
for key, value in loaded_data.items():
    print(f"\nData from file {key}:")
    print(value[0:5])

