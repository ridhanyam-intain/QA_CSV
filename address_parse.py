import pandas as pd

# Function to parse address
def parse_address(address):
    # Check for NaN or empty values
    if pd.isna(address) or address == '':
        return {
            "area": None,
            "city": None,
            "state": None,
            "pincode": None,
            "country": None
        }

    # Split by ", " to divide the address components
    parts = address.split(", ")
    
    # Initialize variables
    area = None
    city = None
    state = None
    pincode = None
    country = None

    # Country is the last element
    if len(parts) > 0:
        country = parts[-1]
    
    # Check for pincode (6-digit number)
    if len(parts) > 1 and parts[-2].isdigit() and len(parts[-2]) == 6:
        pincode = parts[-2]
        # If pincode exists, adjust indices for other components
        state_index = -3
        city_index = -4
    else:
        # If no pincode, adjust indices for other components
        pincode = None
        state_index = -2
        city_index = -3
    
    # Extract state
    if len(parts) >= abs(state_index):
        state = parts[state_index]
    
    # Extract city
    if len(parts) >= abs(city_index):
        city = parts[city_index]
    
    # Area is everything else
    if len(parts) > abs(city_index):
        area = ", ".join(parts[:city_index])

    return {
        "area": area,
        "city": city,
        "state": state,
        "pincode": pincode,
        "country": country
    }

# # Sample addresses
# addresses = [
#     "Jayendra Colony, Ward 183, Zone 14 Perungudi, Sholinganallur, Chennai, Tamil Nadu, 600041, India",
#     "Chennai, Tamil Nadu, 600001, India",
#     "Vadodara Rural Taluka, Vadodara, Gujarat, India"
# ]

combined_data = pd.read_csv('combined_data.csv')

print(combined_data.head())

# loop throuth the combined_data and parse the address
for index, row in combined_data.iterrows():
    # Handle Restaurant address
    restaurant_address = row['Restaurant_address']
    parsed_restaurant_address = parse_address(restaurant_address)
    
    # Update restaurant address fields
    for field in ['area', 'city', 'state', 'pincode', 'country']:
        combined_data.at[index, f'Restaurant_{field}'] = parsed_restaurant_address[field]

    # Handle Delivery address
    delivery_address = row['Delivery_address']
    parsed_delivery_address = parse_address(delivery_address)
    
    # Update delivery address fields
    for field in ['area', 'city', 'state', 'pincode', 'country']:
        combined_data.at[index, f'Delivery_{field}'] = parsed_delivery_address[field]

# save the updated combined_data to a new CSV file
combined_data.to_csv('data_with_address.csv', index=False)

