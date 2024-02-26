# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:34:34 2024

@author: valer
"""


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, tukey_hsd

# File paths and titles
file_paths = [r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\resultsexp_speeds3.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\resultsexp_speeds3.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsexp_speeds.csv", 
              ]
titles = ['DW-1', 'DW-2', 'Alg-1', 'Alg-2', 'Alg-3' , 'Alg-4']

datasets = {}
for title, file_path in zip(titles, file_paths):
    data = pd.read_csv(file_path)['Speed']
    data /= 100 
    datasets[title] = data[(data <= 0.05) & (data>0.015)]

for title, data in datasets.items():
    print(f"Number of data points in {title}: {len(data)}")
# Create a boxplot with the mean
plt.figure(figsize=(10, 6))
# Define custom colors for each boxplot
colors = ['#a0ced9', '#89CFF0', '#228B22', '#54780A', '#014421', '#12372A']
# Create a boxplot with the mean  
plt.figure(figsize=(10, 6))
bp = plt.boxplot(datasets.values(), labels=datasets.keys(), patch_artist=True, showmeans=False, 
                 meanprops=dict(marker='^', markerfacecolor='grey', markeredgecolor='black'),
                 medianprops=dict(color= 'black'))

# Set colors for each boxplot
for box, color in zip(bp['boxes'], colors):
    box.set_facecolor(color)

#plt.title('Boxplot of Average Velocities with Mean')
plt.ylabel('Velocity [m/s]')
plt.ylim(0, 0.04)
plt.grid(True)
plt.savefig(r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\allspeeds_box_with4.png', dpi = 800)
plt.show()


group0 = datasets['DW1']
group1 = datasets['DW2']
group2 = datasets['Alg1']
group3 = datasets['Alg2']
group4 = datasets['Alg3']
group5 = datasets['Alg4']
res = tukey_hsd(group0, group1, group2, group3, group4, group5)
print(res)

# Parse the results into a DataFrame
# Prepare results for DataFrame
statistics = res.statistic
p_values = res.pvalue
CI_lower = res._ci[0]
CI_upper = res._ci[1]
results_data = []
for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        comparison = f"({titles[i]} - {titles[j]})"
        statistic = statistics[i, j]
        p_value = p_values[i, j]
        lower_ci = CI_lower[i, j]
        upper_ci = CI_upper[i, j]
        results_data.append([comparison, statistic, p_value, lower_ci, upper_ci])

# Create DataFrame
columns = ['Comparison', 'Statistic', 'p-value', 'Lower CI', 'Upper CI']
result_df = pd.DataFrame(data=results_data, columns=columns)
result_df.to_csv(r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\tukeyhsdresults_allspeeds.csv")