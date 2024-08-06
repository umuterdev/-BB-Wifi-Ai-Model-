import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path to the combined CSV file
file_path = r'C:\Users\erumu\OneDrive - MEF Üniversitesi\Desktop\features\combined.csv'

# Load the CSV file into a DataFrame with specified encoding
try:
    df = pd.read_csv(file_path, encoding='ansi')  # Use 'ansi' encoding
except UnicodeDecodeError as e:
    print(f"Error reading the CSV file: {e}")
    exit(1)

# Select only numeric columns for correlation matrix calculation
numeric_df = df.select_dtypes(include=[float, int])

# Check if there are multiple numeric columns
if numeric_df.shape[1] > 1:
    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Print the correlation matrix
    print(correlation_matrix)

    # Save the correlation matrix to a CSV file
    correlation_matrix.to_csv('correlation_matrix_output.csv')

    # Plot the correlation matrix using a heatmap with increased figure size
    plt.figure(figsize=(19.2, 10.8))  # Set the figure size to 1920x1080 pixels
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Heatmap')

    # Save the heatmap to a file
    plt.savefig('correlation_matrix_heatmap.png')

    # Show the heatmap
    plt.show()
else:
    print("The DataFrame does not contain multiple numeric columns for correlation calculation.")