import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Read the CSV file
df = pd.read_csv('your_file_path', encoding='ANSI', delimiter=';')

# Remove rows with 'Bilinmiyor' values
df = df[~df.isin(['Bilinmiyor']).any(axis=1)]

# Remove rows where LOCATION is 'IETT Otobus'
df = df[df['LOCATION'] != 'IETT Otobus']

# Drop unnecessary columns
df = df.drop(columns=['LOCATION_VALUE', 'LOCATION_CODE', 'NEIGHBORHOOD_NAME'])

# Group by LOCATION and sum the NUMBER_OF_DISTINCT_USERS
df_grouped = df.groupby('LOCATION')['NUMBER_OF_DISTINCT_USERS'].sum().reset_index()

# Multiply the NUMBER_OF_DISTINCT_USERS by 1.2401 and round the result
df_grouped['NUMBER_OF_DISTINCT_USERS'] = (df_grouped['NUMBER_OF_DISTINCT_USERS'] * 1.2401).round()

# Sort the DataFrame by NUMBER_OF_DISTINCT_USERS in descending order
df_grouped = df_grouped.sort_values(by='NUMBER_OF_DISTINCT_USERS', ascending=False)

# Save the grouped DataFrame to a CSV file
# Save the grouped DataFrame to a CSV file in the csvfiles folder
df_grouped.to_csv('your_file_path', index=False)

# Set up the plot
plt.figure(figsize=(15, 10))

# Create a horizontal bar chart with the Y-axis in the order of the sorted LOCATION column
plt.barh(df_grouped['LOCATION'], df_grouped['NUMBER_OF_DISTINCT_USERS'], color='skyblue')

# Add labels and title
plt.xlabel('Number of Distinct Users')
plt.ylabel('Location')
plt.title('Number of Distinct Users by Location in Istanbul')

# Show grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Ensure the correct display of Turkish characters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Save the plot
plt.tight_layout()
save_path = 'your_file_path'
plt.savefig(save_path, format='png')

# Optionally display the plo
#
# t
plt.show()
