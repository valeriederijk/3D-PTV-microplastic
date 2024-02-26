# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:23:15 2024

@author: valer
"""

import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

# File paths and titles
file_paths = [
    r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\resultsexp_speeds3.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\resultsexp_speeds3.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\resultsexp_speeds.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\resultsexp_speeds.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\resultsexp_speeds.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsexp_speeds.csv"
]
titles = ['DW1', 'DW2', 'Alg1', 'Alg2', 'Alg3', 'Alg4']

# Load datasets from CSV files
datasets = {}
for title, file_path in zip(titles, file_paths):
    data = pd.read_csv(file_path)['Speed']
    data /= 100
    datasets[title] = data[(data <= 0.05) & (data>0.02)]

# Initialize lists to store results
results = []

# Perform t-tests and Mann-Whitney U tests between all combinations of datasets
for i in range(len(titles)):
    for j in range(i+1, len(titles)):
        title1 = titles[i]
        title2 = titles[j]
        data1 = datasets[title1]
        data2 = datasets[title2]

        # Perform t-test
        t_statistic, t_p_value = ttest_ind(data1, data2)
        t_effect_size = (data1.mean() - data2.mean()) / ((data1.std() + data2.std()) / 2)

        # Perform Mann-Whitney U test
        u_statistic, u_p_value = mannwhitneyu(data1, data2)
        u_effect_size = 2 * u_statistic / (len(data1) * len(data2)) - 1

        # Append results to the list
        results.append([title1, title2, t_statistic, t_p_value, t_effect_size, u_statistic, u_p_value, u_effect_size])

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['Dataset 1', 'Dataset 2', 'T-statistic', 'T-p-value', 'T-effect size', 'U-statistic', 'U-p-value', 'U-effect size'])


# Save the DataFrame to an Excel file
excel_file_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\statistics_all_groups_all_speeds.xlsx"
results_df.to_excel(excel_file_path, index=False)

print("Results saved to", excel_file_path)
