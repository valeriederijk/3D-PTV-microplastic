# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:54:26 2024

@author: valer
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from CSV files for all experiments
experiment_files = [
    r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\resultsexp3.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\resultsexp.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\resultsexp.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\resultsexp.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\resultsexp.csv",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsexp.csv"
]

dfs_exp = [pd.read_csv(file) for file in experiment_files]
dfs_exp = [df[(df['Average Velocity'] > 0.02) & (df['Average Velocity'] <= 0.1)] for df in dfs_exp]
#dfs_2d = [df[(df['Average Speed'] > 0.02) & (df['Average Speed'] <= 0.1)]

# Load 2d velocity data from Excel files for each experiment
data_files = [
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\updated2dspeeds.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\updated2dspeeds.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\updated2dspeeds.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\updated2dspeeds.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\updated2dspeeds.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\updated2dspeeds.xlsx"
]

dfs_2d = [pd.read_excel(file) for file in data_files]
dfs_2d = [df[(df['Average Speed'] > 0) & (df['Average Speed'] <= 0.05)] for df in dfs_2d]

# Define colors for each experiment
experiment_colors = ['#a0ced9', '#89CFF0', '#228B22', '#54780A', '#014421', '#12372A']
titles = ['DW-1', 'DW-2', 'Alg-1', 'Alg-2', 'Alg-3', 'Alg-4']

# Create a 3x2 subplot grid
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Iterate through experiments and plot density distribution
for i, (df_exp, df_2d, color, title) in enumerate(zip(dfs_exp, dfs_2d, experiment_colors, titles)):
    row = i // 2
    col = i % 2
    ax = axs[row, col]

    # Plot density distribution with KDE for 3D speed
    sns.kdeplot(data=df_exp['Average Velocity'], fill=True, label='Experiment', color=color, common_norm=True, ax=ax)

    # Plot density distribution with KDE for 2D speed
    sns.kdeplot(data=df_2d['Average Speed'], fill=True, label='2D', color='#573584', common_norm=True, ax=ax)

    # Plot mean dashed lines
    mean_speed_exp = df_exp['Average Velocity'].mean()
    mean_speed_2d = df_2d['Average Speed'].mean()
    ax.axvline(mean_speed_exp, color=color, linestyle='dashed', linewidth=2)
    ax.axvline(mean_speed_2d, color='#573584', linestyle='dashed', linewidth=2)

    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Density')
    ax.set_title(title)
    
    # Set x-axis limits
    ax.set_xlim(0, 0.06)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\2d_vs_3d_speed_experiments_updated.png', dpi=800)

# Display the plot
plt.show()
