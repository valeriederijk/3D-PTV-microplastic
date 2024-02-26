# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:48:19 2024

@author: valer
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# Assuming you have 6 CSV files named 'file1.csv', 'file2.csv', ..., 'file6.csv'
file_paths = [r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\particle_analyzed\stokesvelocity.xlsx",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\particle_analyzed\stokesvelocity.xlsx", 
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\particle_analyzed\stokesvelocity.xlsx",
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\particle_analyzed\stokesvelocity.xlsx", 
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\particle_analyzed\stokesvelocity.xlsx", 
              r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\stokesvelocity.xlsx"]

titles = ['DW-1', 'DW-2', 'Alg-1', 'Alg-2', 'Alg-3', 'Alg-4']
# Create a figure with 2x3 subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey=True, sharex = True)

# Flatten the 2D array of axes to simplify indexing
axes = axes.flatten()

for i, (file_path, title) in enumerate(zip(file_paths, titles)):
    # Read data from Excel file
    df = pd.read_excel(file_path, sheet_name='Sheet1')  #
    
    # Create a distribution plot for the equivalent diameters
    ax =sns.histplot(data=df, x='d_eq_m', kde=False, color='#B4B4B8', ax=axes[i], bins=15, stat='density' ) # Adjust bins as needed
    sns.kdeplot(data=df, x="d_eq_m", color='black', ax=ax)
    # Set subplot title
    axes[i].set_title(title)
    
    # Set axis labels
    axes[i].set_xlabel('Equivalent Diameter [m]')
    axes[i].set_ylabel('Frequency')
    axes[i].set_xlim(0, 0.0018)
    axes[i].set_ylim(0, 5500)
    # Format x-axis tick labels in scientific notation with 'e-3' unit
    axes[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[i].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[i].xaxis.offsetText.set_visible(True)
    
save_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\allparticlesdensity.png"

# Save the figure
# Adjust layout to prevent overlapping titles
plt.tight_layout()
plt.savefig(save_path, dpi = 800)
# Show the plots
plt.show()
