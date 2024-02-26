# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:09:12 2024

@author: valer
"""

import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np

# Load coordinates from CSV file
df_trajectories = pd.read_csv(r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\3d_trajectories_trial.csv")

# Initialize figure
fig = go.Figure()

# Generate a colormap with unique colors for each particle
colormap = plt.cm.rainbow(np.linspace(0, 1, len(df_trajectories['Particle'].unique())))

# Initialize variables to store overall minimum coordinates
min_x = df_trajectories['X'].min()
min_y = df_trajectories['Y'].min()

coordinates_df = pd.DataFrame(columns=['Particle', 'X', 'Y', 'Z'])

for particle_index, data in df_trajectories.groupby('Particle'):
    x_coords = data['X'].values - min_x  # Normalize x coordinates
    y_coords = data['Y'].values - min_y  # Normalize y coordinates
    z_coords = data['Z'].values

    particle_df = pd.DataFrame({
        'Particle': [particle_index] * len(x_coords),
        'X': x_coords,
        'Y': y_coords,
        'Z': z_coords
    })
    coordinates_df = pd.concat([coordinates_df, particle_df])

    # Plotting 3D trajectories with unique colors for each particle
    color = colormap[particle_index % len(colormap)]
    line_color = np.tile(color, (len(x_coords), 1))
    scatter = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines', name=f'Particle {particle_index}',
        line=dict(color=line_color, width=3)
    )
    fig.add_trace(scatter)

# Set layout
fig.update_layout(scene=dict(
    aspectmode="data",  # Set aspect mode to 'data' for custom aspect ratios
    xaxis=dict(title='', tickmode='linear', tick0=0, dtick=2),
    yaxis=dict(title='', tickmode='linear', tick0=0, dtick=2),
    zaxis=dict(title='', tickmode='linear', tick0=0, dtick=2)
), title='Particle 3D Trajectories')

# Save the interactive plot
fig.write_html(r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\3d_trajectories_trial2.html")

# Show the interactive plot
fig.show()
