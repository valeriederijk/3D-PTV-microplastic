# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:16:40 2024

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

# Load theoretical distributions from Excel files for each experiment
theoretical_files = [
    r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\combined_settling_velocity_results.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\combined_settling_velocity_results.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\combined_settling_velocity_results.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\combined_settling_velocity_results.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\combined_settling_velocity_results.xlsx",
    r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\combined_settling_velocity_results.xlsx"
]

dfs_theoretical = [pd.read_excel(file) for file in theoretical_files]

# Define columns to plot from theoretical files
columns_to_plot = ['v_dioguardi', 'v_swamee', 'v_ganser', 'v_dietrich', 'v_stokes']
column_names = ['Dioguardi', 'Swamee', 'Ganser', 'Dietrich', 'Stokes']

# Define colors for each column
experiment_colors = ['#a0ced9', '#89CFF0', '#228B22', '#54780A', '#014421', '#12372A']
column_colors = ['#573584', '#F8766D', '#00BA38', '#619CFF', '#F564E3']
titles = ['DW-1', 'DW-2', 'Alg-1', 'Alg-2', 'Alg-3', 'Alg-4']
# Create a 3x2 subplot grid
fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

# Iterate through experiments and plot density distribution for theoretical data
for i, (df_exp, df_theoretical, color, title) in enumerate(zip(dfs_exp, dfs_theoretical, experiment_colors, titles)):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    sns.kdeplot(data=df_exp['Average Velocity'], fill=True, label='Experiment', color=color, common_norm=False, ax=ax)
    
    
    for col_to_plot, col_name, col_color in zip(columns_to_plot, column_names, column_colors):
        sns.kdeplot(data=df_theoretical[col_to_plot], fill=False, label=col_name, color=col_color, common_norm=False, ax=ax)
        
    
    
    # Plot mean dashed lines
    #mean_speed_exp = df_exp['Average Velocity'].mean()
    #ax.axvline(mean_speed_exp, color='red', linestyle='dashed', linewidth=2)
    
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Density')
    ax.set_title(title)
    # Set x-axis limits
    ax.set_xlim(0, 0.06)
    
axs[0, 0].legend()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\particle_analyzed\theoretical.png', dpi=800)

# Display the plot
plt.show()