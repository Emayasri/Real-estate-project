import pandas as pd

# Define the data as a dictionary with lists of the same length
data = {
    'Square_Feet': [1500, 2000, 2500, 3000, 1200, 1800, 1600, 2200, 2100, 2400, 2800, 3000,
                    1700, 2500, 1500, 1900, 2100, 3000, 2200, 2400, 2600, 2000, 1500, 2800,
                    2400, 2600, 1900, 2300, 2200, 2500, 1800, 2100, 2000, 1500, 1700, 2500,
                    2300, 1800, 2000, 3000, 4000],
    'Bedrooms': [3, 2, 3, 3, 2, 3, 3, 3, 2, 2, 4, 2, 3, 4, 3, 3, 2, 2, 3, 4, 3, 3, 2, 4,
                 3, 3, 2, 4, 3, 3, 2, 3, 4, 4, 3, 3, 2, 3, 4, 3, 4],
    'Location': ['Chennai', 'Coimbatore', 'Chennai', 'Coimbatore', 'Chennai', 'Tiruchirappalli', 'Chennai',
             'Coimbatore', 'Chennai', 'Tiruchirappalli', 'Chennai', 'Coimbatore', 'Chennai', 'Coimbatore',
             'Chennai', 'Tiruchirappalli', 'Chennai', 'Coimbatore', 'Chennai', 'Tiruchirappalli', 'Chennai',
             'Coimbatore', 'Chennai', 'Tiruchirappalli', 'Chennai', 'Coimbatore', 'Chennai', 'Tiruchirappalli',
             'Coimbatore', 'Chennai', 'Tiruchirappalli', 'Chennai', 'Coimbatore', 'Chennai', 'Tiruchirappalli',
             'Chennai', 'Coimbatore', 'Chennai', 'Tiruchirappalli', 'Chennai', 'Coimbatore'],
    'Age_of_Property': [5, 10, 15, 2, 20, 8, 7, 3, 5, 12, 1, 4, 10, 6, 18, 3, 15, 1, 7, 2, 8, 20, 4, 11,
                        5, 3, 10, 6, 12, 4, 14, 5, 10, 18, 5, 3, 10, 4, 8, 12, 5],
    'Price': [5000000, 6000000, 6500000, 800000, 3500000, 5500000, 5200000, 6200000, 5750000, 6800000, 750000,
              7000000, 5400000, 7700000, 3200000, 5900000, 6300000, 8200000, 6200000, 6800000, 7300000, 5700000,
              5100000, 7600000, 6400000, 7100000, 5300000, 7400000, 6600000, 6000000, 4900000, 5750000, 6000000,
              4200000, 4800000, 6700000, 7100000, 4900000, 5300000, 7500000, 7000000]
}

# Check if all lists have the same length
list_lengths = [len(v) for v in data.values()]
if len(set(list_lengths)) > 1:
    print("Error: Lists have different lengths")
else:
    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('real_estate123.csv', index=False)

    print("Dataset created and saved as 'real_estate123.csv'")
