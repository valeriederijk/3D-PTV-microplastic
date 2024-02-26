# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:46:09 2024

@author: valer
"""

import pandas as pd
from scipy.stats import f_oneway, shapiro

# File paths and titles
file_paths = [r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\particle_data_new.xlsx",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\particle_data_new.xlsx",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\particle_data_new.xlsx",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\particle_data_new.xlsx",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\particle_data_new.xlsx",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_data_new.xlsx", 
              ]
titles = ['DW1', 'DW2', 'Alg1', 'Alg2', 'Alg3' , 'Alg4']


# Load datasets from Excel files
datasets = {}
for title, file_path in zip(titles, file_paths):
    data = pd.read_excel(file_path, usecols=['Equivalent Diameter'])
    datasets[title] = data

# Perform normality testing (Shapiro-Wilk test)
normality_results = {}
for title, data in datasets.items():
    _, p_value = shapiro(data)
    normality_results[title] = {'P-value': p_value, 'Normal': p_value > 0.05}

# Perform ANOVA test
f_statistic, p_value = f_oneway(datasets['DW1'], datasets['DW2'], datasets['Alg1'], datasets['Alg2'], datasets['Alg3'], datasets['Alg4'])

# Compute descriptive statistics
descriptive_stats = pd.DataFrame({
    'Title': titles,
    'Mean': [dataset.mean() for dataset in datasets.values()],
    'Standard Deviation': [dataset.std() for dataset in datasets.values()],
    'Min': [dataset.min() for dataset in datasets.values()],
    'Max': [dataset.max() for dataset in datasets.values()],
    'Normality P-value': [normality_results[title]['P-value'] for title in titles],
    'Normal': [normality_results[title]['Normal'] for title in titles]
})

# Define the location to save the results
results_location =  r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\deq_anova.xlsx"

# Create a DataFrame for the ANOVA results
anova_results = pd.DataFrame({
    'F-Statistic': [f_statistic],
    'P-value': [p_value],
    'Reject Null Hypothesis': [p_value < 0.05]
})

# Write results to an Excel file
with pd.ExcelWriter(results_location) as writer:
    descriptive_stats.to_excel(writer, sheet_name='Descriptive_Statistics', index=False)
    anova_results.to_excel(writer, sheet_name='ANOVA_Results', index=False)

# Print confirmation message
print("Results saved to:", results_location)
