# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:21:01 2024

@author: valer
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def load_data(file_path, min_velocity=0.015, max_velocity=0.040):
    df = pd.read_csv(file_path)
    df = df[(df['Average Velocity'] >= min_velocity) & (df['Average Velocity'] <= max_velocity)]
    return df

# Assuming you have 6 CSV files named 'file1.csv', 'file2.csv', ..., 'file6.csv'
file_paths = [r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\resultsexp3.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\resultsexp.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\resultsexp.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\resultsexp.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\resultsexp.csv",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsexp.csv"]
titles = ['DW-1', 'DW-2', 'Alg-1', 'Alg-2', 'Alg-3', 'Alg-4']
# Create a figure with 3x2 subplots

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15), sharex=True)

# Flatten the 2D array of axes to simplify indexing
axes = axes.flatten()

for i, (file_path, title) in enumerate(zip(file_paths, titles)):
    # Load data and filter based on average velocity
    df = load_data(file_path)
    
    # Create QQ plot
    stats.probplot(df['Average Velocity'], dist="norm",  plot=axes[i])
    axes[i].get_lines()[0].set_markerfacecolor('black')
    axes[i].get_lines()[0].set_markeredgecolor('black')
    axes[i].get_lines()[0].set_color('black')
    # Set subplot title
    axes[i].set_title(title)
    
# Adjust layout
plt.tight_layout()

save_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\allparticlesqqplot.png"

# Save the figure
# Adjust layout to prevent overlapping titles
plt.tight_layout()
plt.savefig(save_path, dpi = 800)
# Show the plots
plt.show()
