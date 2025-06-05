"""
This is an example of plotting data from JSON files to
graphs using matplotlib and pandas.

Depenencies:
- matplotlib
- pandas

Data files:
- APLE.json
- TATACOMM.BSE.json
"""

import matplotlib.pyplot as plt
import pandas as pd

df_aple = pd.read_json('APLE.json', orient='index')
df_aple.index.name = 'date'
df_aple.reset_index(inplace=True)

# Convert columns to appropriate data types
df_aple['4. close'] = df_aple['4. close'].astype(float)
df_aple['date'] = pd.to_datetime(df_aple['date'])

# Sort by date
df_aple = df_aple.sort_values('date')

start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2025-06-01')
df_aple_filtered = df_aple[(df_aple['date'] >= start_date) & (df_aple['date'] <= end_date)]


df_tata = pd.read_json('TATACOMM.BSE.json', orient='index')
df_tata.index.name = 'date'
df_tata.reset_index(inplace=True)
# Convert columns to appropriate data types
df_tata['4. close'] = df_tata['4. close'].astype(float)
df_tata['date'] = pd.to_datetime(df_tata['date'])

# Sort by date
df_tata = df_tata.sort_values('date')
df_tata_filtered = df_tata[(df_tata['date'] >= start_date) & (df_tata['date'] <= end_date)]

# Single Plot
plt.figure(figsize=(12, 6))  
plt.plot(df_aple['date'], df_aple['4. close'])
plt.title('Apple Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)  # Add a grid for easier reading of values
plt.savefig('aple_only.png')
# plt.show()

plt.figure(figsize=(14, 7)) # Set the figure size for better readability

# Plot TATA data
plt.plot(df_tata_filtered['date'], df_tata_filtered['4. close'], label='TATA Close Price', marker='o', markersize=4, linestyle='-')

# Plot APLE data
plt.plot(df_aple_filtered['date'], df_aple_filtered['4. close'], label='APLE Close Price', marker='x', markersize=4, linestyle='--')

# Add titles and labels
plt.title('Stock Close Prices Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price', fontsize=12)

# Add a legend to distinguish the lines
plt.legend(fontsize=10)

# Add grid for better readability
plt.grid(True, linestyle=':', alpha=0.7)

# Improve date formatting on x-axis
plt.gcf().autofmt_xdate()

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.savefig('aplemat.jpg')
# plt.show()

