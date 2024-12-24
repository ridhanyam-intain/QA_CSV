# import pandas as pd

# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')

# data = pd.concat([train_data, test_data], ignore_index=True)

# data.to_csv('data.csv', index=False)

#-----------------------------------------------------------------------------------------------

# Read data.csv and display the first 5 rows
# data = pd.read_csv('data.csv')
# print(data.head())
# # show all columns
# print(data.columns)
# # Show the shape of the data
# print(data.shape)

#-----------------------------------------------------------------------------------------------

import pandas as pd
from geopy.geocoders import Nominatim
import time
from tqdm import tqdm
import os

# Read data.csv
data = pd.read_csv('data.csv')

# Set the start and end indices
START_INDEX = 8500
END_INDEX = 10000

# Slice the data to only process records between START_INDEX and END_INDEX
data = data.iloc[START_INDEX:END_INDEX].copy()
# Display the first 5 rows of the sliced data
print(data.head())

# Initialize geolocator
geolocator = Nominatim(user_agent="geoapi")

def get_address(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        return location.address if location else None
    except Exception as e:
        print(f"Error: {e}")
        return None

def is_valid_coordinates(lat, lon):
    # Check if coordinates are too close to 0,0 or have invalid values
    if abs(lat) < 1 or abs(lon) < 1:
        return False
    return True

# Process data in batches
BATCH_SIZE = 10
OUTPUT_FILE = f'data_with_areas_{START_INDEX}_{END_INDEX}.csv'

# Create empty columns for areas
data['Restaurant_area'] = None
data['Delivery_area'] = None

# Check if output file exists and load previous progress
if os.path.exists(OUTPUT_FILE):
    processed_data = pd.read_csv(OUTPUT_FILE)
    # Find the last processed index relative to the sliced data
    last_processed = len(processed_data)
    print(f"Resuming from index {last_processed}")
else:
    last_processed = 0
    # Create the file with headers
    data.head(0).to_csv(OUTPUT_FILE, index=False)

print(len(data))
print(last_processed)
# Process in batches with progress bar
for i in tqdm(range(last_processed, len(data), BATCH_SIZE)):
    batch = data.iloc[i:i + BATCH_SIZE].copy()
    
    # Process each row in the batch
    for idx, row in batch.iterrows():
        # Validate restaurant coordinates
        if is_valid_coordinates(row['Restaurant_latitude'], row['Restaurant_longitude']):
            batch.at[idx, 'Restaurant_area'] = get_address(
                row['Restaurant_latitude'], 
                row['Restaurant_longitude']
            )
            time.sleep(1.1)
        else:
            batch.at[idx, 'Restaurant_area'] = None
        
        # Validate delivery coordinates
        if is_valid_coordinates(row['Delivery_location_latitude'], row['Delivery_location_longitude']):
            batch.at[idx, 'Delivery_area'] = get_address(
                row['Delivery_location_latitude'], 
                row['Delivery_location_longitude']
            )
            time.sleep(1.1)
        else:
            batch.at[idx, 'Delivery_area'] = None
    
    # Append this batch to the CSV file
    batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    print(f"Batch processed and saved: rows {i} to {min(i + BATCH_SIZE, len(data))}")

print("Processing complete! All results saved to 'data_with_areas.csv'")
