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
import requests
import time
from tqdm import tqdm
import os

# Read data.csv
data = pd.read_csv('C:/Users/Ridhanya/Documents/Intain/Samy/NewData/df_train_with_clusters_revised_v6.csv')

# Set the start and end indices
START_INDEX = 40000
END_INDEX = 50000

# Slice the data to only process records between START_INDEX and END_INDEX
data = data.iloc[START_INDEX:END_INDEX].copy()
# print(data.columns)

# Replace Nominatim initialization with Mapbox API key
API_KEY = "pk.eyJ1IjoicmlkaGFueWEiLCJhIjoiY201bGN1bGNtMHRrNzJpc2NtdWxyOHFldyJ9.oEi3bOuNGwJKyZZEeHpyjQ"


# Replace get_address function
def get_address(lat, lon):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {
        "access_token": API_KEY,
        "types": "address",
        "limit": 1
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get("features", [])
            if results:
                # return results[0].get("place_name", None)
                return results[0]  # Return the full result object
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def is_valid_coordinates(lat, lon):
    # Check if coordinates are too close to 0,0 or have invalid values
    if abs(lat) < 1 or abs(lon) < 1:
        return False
    return True

def parse_address_details(feature):
    if not feature:
        return None
    
    context = {c["id"].split(".")[0]: c["text"] for c in feature.get("context", [])}
    
    return {
        "street": feature.get("text"),
        "full_address": feature.get("place_name"),
        "neighborhood": context.get("neighborhood"),
        "postcode": context.get("postcode"),
        "locality": context.get("locality"),
        "city": context.get("place"),
        "district": context.get("district"),
        "state": context.get("region"),
        "state_code": context.get("region_short_code"),
        "country": context.get("country"),
        "country_code": context.get("country_short_code"),
        "coordinates": feature.get("center")
    }

if __name__ == "__main__":
    BATCH_SIZE = 500
    OUTPUT_FOLDER = 'C:/Users/Ridhanya/Documents/Intain/Samy/NewData'
    OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'data_{}_{}.csv'.format(START_INDEX, END_INDEX))

    # Add full mapbox response columns along with other components
    address_components = [
        'street', 'full_address', 'neighborhood', 'postcode', 
        'locality', 'city', 'district', 'state', 'state_code',
        'country', 'country_code', 'coordinates', 'mapbox_response'
    ]
    
    for component in address_components:
        data[f'Restaurant_{component}'] = None
        data[f'Delivery_{component}'] = None

    # Check if output file exists and load previous progress
    if os.path.exists(OUTPUT_FILE):
        processed_data = pd.read_csv(OUTPUT_FILE)
        last_processed = len(processed_data)
        print(f"Resuming from index {last_processed}")
    else:
        last_processed = 0
        data.head(0).to_csv(OUTPUT_FILE, index=False)

    # Process in batches with progress bar
    for i in tqdm(range(last_processed, len(data), BATCH_SIZE)):
        batch = data.iloc[i:i + BATCH_SIZE].copy()
        
        # Process each row in the batch
        for idx, row in batch.iterrows():
            # Process restaurant address
            if is_valid_coordinates(row['restaurant_latitude'], row['restaurant_longitude']):
                result = get_address(
                    row['restaurant_latitude'], 
                    row['restaurant_longitude']
                )
                if result:
                    batch.at[idx, 'Restaurant_mapbox_response'] = str(result)
                    details = parse_address_details(result)
                    if details:
                        for component, value in details.items():
                            batch.at[idx, f'Restaurant_{component}'] = str(value)
                time.sleep(0.1)
            
            # Process delivery address
            if is_valid_coordinates(row['delivery_location_latitude'], row['delivery_location_longitude']):
                result = get_address(
                    row['delivery_location_latitude'], 
                    row['delivery_location_longitude']
                )
                if result:
                    batch.at[idx, 'Delivery_mapbox_response'] = str(result)
                    details = parse_address_details(result)
                    if details:
                        for component, value in details.items():
                            batch.at[idx, f'Delivery_{component}'] = str(value)
                time.sleep(0.1)
        
        # Append this batch to the CSV file
        batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print(f"Batch processed and saved: rows {i} to {min(i + BATCH_SIZE, len(data))}")

    print("Processing complete! All results saved")
