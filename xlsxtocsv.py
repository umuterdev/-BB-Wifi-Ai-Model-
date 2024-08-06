import os
import pandas as pd
from zipfile import BadZipFile
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

# We can convert multiple Excel files to CSV format using the openpyxl library
# Name of the file that contains our xlsx files is 'csv_files'. Change it if you have a different name
csv_files_folder = 'csv_files'

# Listing the files in the folder
files = os.listdir(csv_files_folder)

# In case there are any files that could not be converted, we will store them in a list
failed_files = []

# Using a for loop to iterate over the files
for file in files:
    # Define the full file path
    file_path = os.path.join(csv_files_folder, file)

    # Check if the file is not already a CSV file
    if not file.endswith('.csv'):
        try:
            # Change it if you have a different file format
            if file.endswith('.xlsx'):
                # Check if the file is a valid Excel file
                load_workbook(file_path)

                # Read the file into a DataFrame using the openpyxl engine
                df = pd.read_excel(file_path, engine='openpyxl')

                # Define the new CSV file path
                csv_file_path = os.path.splitext(file_path)[0] + '.csv'

                # You can change the encoding if you have a different one preferred
                df.to_csv(csv_file_path, index=False, encoding='utf-8')

                # This part is optional. If you want to delete the original file, you can uncomment the line below
                #os.remove(file_path)

                print(f"Converted: {file_path} to {csv_file_path}")
            else:
                print(f"Skipping: {file_path} is not an .xlsx file.")
        except (BadZipFile, InvalidFileException, ValueError) as e:
            print(f"Error: {file_path} is not a valid Excel file and cannot be read. Exception: {e}")
            failed_files.append(file_path)

print("All valid files have been converted to CSV format.")
if failed_files:
    print("The following files could not be converted:")
    for failed_file in failed_files:
        print(failed_file)