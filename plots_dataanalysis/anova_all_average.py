# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:51:01 2024

@author: valer
"""

import pandas as pd
from scipy.stats import f_oneway, tukey_hsd

# File paths and titles
# File paths and titles
file_paths = [r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\resultsexp_speeds3.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\resultsexp_speeds3.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsexp_speeds.csv", 
              ]
titles = ['DW1', 'DW2', 'Alg1', 'Alg2', 'Alg3' , 'Alg4']

# Load datasets from CSV files
datasets = {}
for title, file_path in zip(titles, file_paths):
    data = pd.read_csv(file_path)['Speed']
    data /= 100 
    datasets[title] = data[(data <= 0.05) & (data>0.02)]

# Perform ANOVA test
f_statistic, p_value = f_oneway(datasets['DW1'], datasets['DW2'], datasets['Alg1'], datasets['Alg2'], datasets['Alg3'], datasets['Alg4'])

# Compute descriptive statistics
descriptive_stats = pd.DataFrame({
    'Title': titles,
    'Mean': [dataset.mean() for dataset in datasets.values()],
    'Standard Deviation': [dataset.std() for dataset in datasets.values()],
    'Min': [dataset.min() for dataset in datasets.values()],
    'Max': [dataset.max() for dataset in datasets.values()]
})

# Define the location to save the results
results_location =  r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\anova_results_allspeeds.csv"

# Create a DataFrame for the ANOVA results
anova_results = pd.DataFrame({
    'F-Statistic': [f_statistic],
    'P-value': [p_value],
    'Reject Null Hypothesis': [p_value < 0.05]
})

# Write results to a CSV file
with open(results_location, 'w') as f:
    descriptive_stats.to_csv(f, index=False)
    f.write('\n\n')
    anova_results.to_csv(f, index=False)

# Print confirmation message
print("Results saved to:", results_location)
