# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:41:12 2024

@author: isaac
"""

import pandas as pd
from netCDF4 import Dataset
import os

# Open the NetCDF file
filename = "IO/GR_processed_connectivity_matrices.nc"
dataset = Dataset(filename, mode='r')

# Inspect the dataset to find the variable needed
print("Available variables:", dataset.variables.keys())

# Replace 'variable_name' with the name of the variable you want to extract
variable_name = 'gen_dist_mod_cv'
data = dataset.variables[variable_name][:]

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Define the full path to save the CSV file inside the 'IO' folder
csv_filename = os.path.join('IO', 'IO_gen_dist_mod_cv_connectivity_matrix.csv')

# Save the DataFrame as a CSV without row and column headers
df.to_csv(csv_filename, index=False, header=False)

print(f"CSV saved as {csv_filename}")

dataset.close()
