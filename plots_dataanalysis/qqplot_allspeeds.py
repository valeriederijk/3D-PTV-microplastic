# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:16:36 2024

@author: valer
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def load_data(file_path, min_velocity=0.015, max_velocity=0.040):
    df = pd.read_csv(file_path)
    df['Speed'] /= 100  # Divide average velocity by 100
    df = df[(df['Speed'] >= min_velocity) & (df['Speed'] <= max_velocity)]
    return df

# Assuming you have 6 CSV files named 'file1.csv', 'file2.csv', ..., 'file6.csv'
file_paths = [r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\resultsexp_speeds3.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\resultsexp_speeds3.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\resultsexp_speeds.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsexp_speeds.csv"]
titles = ['DW-1', 'DW-2', 'Alg-1', 'Alg-2', 'Alg-3', 'Alg-4']
# Create a figure with 3x2 subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15), sharex=True)

# Flatten the 2D array of axes to simplify indexing
axes = axes.flatten()
results = []
previous_group = None 

for i, (file_path, title) in enumerate(zip(file_paths, titles)):
    # Load data and filter based on average velocity
    df = load_data(file_path)
    
    # Create QQ plot
    stats.probplot(df['Speed'], dist="norm",  plot=axes[i])
    axes[i].get_lines()[0].set_markerfacecolor('black')
    axes[i].get_lines()[0].set_markeredgecolor('black')
    axes[i].get_lines()[0].set_color('black')
    # Set subplot title
    axes[i].set_title(title)
    
    # Kolmogorov-Smirnov test for normality
    ks_statistic, ks_p_value = stats.kstest(df['Speed'], 'norm')
    
    # Levene test for equality of variances
    if i > 0:  # Perform Levene test for all groups except the first one
        levene_statistic, levene_p_value = stats.levene(df['Speed'], previous_group['Speed'])
        results.append({'Group 1': titles[i-1], 'Group 2': title, 'Levene Test Statistic': levene_statistic, 'p-value': levene_p_value})
    
    # Store the current group's data for comparison in the next iteration
    previous_group = df
    
    results.append({'Title': title, 'Kolmogorov-Smirnov Test Statistic': ks_statistic, 'p-value': ks_p_value})

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)


# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
excel_file_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\kolmogorov_smirnov_results_allspeeds.xlsx"
results_df.to_excel(excel_file_path, index=False)
    
# Adjust layout
plt.tight_layout()

save_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\allparticleallspeedsqqplot.png"

# Save the figure
# Adjust layout to prevent overlapping titles
plt.tight_layout()
plt.savefig(save_path, dpi = 800)
# Show the plots
plt.show()
