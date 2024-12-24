# Combine data_with_areas.csv, data_with_areas_3200_3300.csv, data_with_areas_3300_3400.csv, data_with_areas_3400_3500.csv, data_with_areas_3500_4000.csv, data_with_areas_4000_5000.csv, data_with_areas_5000_6000.csv, data_with_areas_6000_8000.csv, data_with_areas_8000_8260.csv, data_with_areas_8260_8500.csv, data_with_areas_8500_10000.csv

import pandas as pd

# # Read all CSV files
# files = ['data_with_areas.csv', 'data_with_areas_3200_3300.csv', 'data_with_areas_3300_3400.csv', 'data_with_areas_3400_3500.csv','data_with_areas_3500_4000.csv', 'data_with_areas_4000_5000.csv', 'data_with_areas_5000_6000.csv', 'data_with_areas_6000_8000.csv', 'data_with_areas_8000_8260.csv', 'data_with_areas_8260_8500.csv', 'data_with_areas_8500_10000.csv']
# dataframes = [pd.read_csv(file) for file in files]

# # Concatenate all dataframes
# combined_data = pd.concat(dataframes, ignore_index=True)

# # Save the combined data to a new CSV file
# combined_data.to_csv('combined_data.csv', index=False)


# Read the combined data
combined_data = pd.read_csv('combined_data.csv')

# Display the first 5 rows of the combined data
print(combined_data.head())

print(combined_data.columns)

