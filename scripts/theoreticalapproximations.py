# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:13:50 2024

@author: valer
"""

import numpy as np
import pandas as pd

# Load data from the Excel file
file_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\particle_data_new.xlsx"
D = pd.read_excel(file_path)

# Convert Deq to meters
D['d_eq_m'] = D['Equivalent Diameter'] / 100

# Constants
rho = 1200  # kg/m^3 PVC sphere density
ni = 1.002e-6  # m^2/s water kinematic viscosity
rho_w = 1032  # kg/m^3 water density
g = 9.81  # m/s^2 gravitational acceleration

# Define the v_stokes function
def v_stokes(d_eq_m, v_0, ni, rho, rho_w, g):
    v_i = v_0
    Re_0 = d_eq_m * v_0 / ni
    CD_0 = 24 / Re_0 + 4 / np.sqrt(Re_0) + 0.4  # Stokes
    Re_i = Re_0
    CD_i = CD_0
    while True:
        Re_i = d_eq_m * v_i / ni
        CD_i = 24 / Re_i + 4 / np.sqrt(Re_i) + 0.4
        v_i1 = np.sqrt(4 / 3 * d_eq_m / CD_i * ((rho - rho_w) / rho_w) * g)
        RSS = (v_i - v_i1) ** 2
        v_i = v_i1
        if RSS < 1e-100:
            break
    return v_i, Re_i

# Define the Dietrich settling velocity function
def dietrich_settling_velocity(dp, rho, mu, rho_w, deq_file, CSF, P):
    # Load Deq data from another Excel file
    deq_data = pd.read_excel(deq_file)
    
    # Function to calculate settling velocity using Dietrich formula

    def calculate_D_ast(rho_p, rho_w, g, deq, ni):
        D_ast = ((rho_p - rho_w) * g * (deq ** 3)) / (rho_w * (ni ** 2))
        return D_ast
    # Function to calculate dimensionless parameter W_ast
    def calculate_W_ast(R3, R1, R2):
        W_ast = (R3 ** 3) *( 10 ** (R1 + R2))
        return W_ast

    # Function to calculate R1
    def calculate_R1(logD):
        return -3.76715 + 1.92944 * logD - 0.09815 * (logD ** 2.0) - 0.00575 * (logD ** 3.0) + 0.00056 * (logD ** 4.0)

    # Function to calculate R2
    def calculate_R2(logD, CSF):
        return (np.log(1 - (1 - CSF) / 0.85)) - (1 - CSF) ** 2.3 * np.tanh(logD - 4.6) + 0.3 * (0.5 - CSF) * (1 - CSF) ** 2.0 * (logD - 4.6)

    # Function to calculate R3
    def calculate_R3(logD, CSF, P):
        return (0.65 - (CSF / 2.83 * np.tanh(logD - 4.6))) ** (1 + (3.5 - P) / 2.5)

    
    # Function to calculate settling velocity using Dietrich formula
    def calculate_W(dp, rho, rho_w, mu, W_ast):
        g = 9.81  # acceleration due to gravity (m/s^2)
        W = ((rho - rho_w / rho_w) * g *ni * W_ast) ** (1/3)
        return W

    # Add calculated column for sphere equivalent diameter in m
    deq_data['d_eq_m'] = deq_data['Equivalent Diameter'] / 100

    # Calculate Stokes velocity for each sphere
    deq_data[['v_stk', 'Re']] = deq_data.apply(lambda row: pd.Series(v_stokes(row['d_eq_m'], 0.02, ni, rho, rho_w, g)), axis=1, result_type='expand')

    # Calculate Dietrich settling velocity using Dietrich formula
    deq_data['D_ast'] = calculate_D_ast(rho, rho_w, g, deq_data['d_eq_m'], ni)
    # Calculate dimensionless parameters
    deq_data['R1'] = calculate_R1(np.log(deq_data['D_ast']))
    deq_data['R2'] = calculate_R2(np.log(deq_data['D_ast']), deq_data['CSF'])
    deq_data['R3'] = calculate_R3(np.log(deq_data['D_ast']), deq_data['CSF'], P)
    deq_data['W_ast'] = calculate_W_ast(deq_data['R3'], deq_data['R1'], deq_data['R2'])
    deq_data['W'] = calculate_W(deq_data['d_eq_m'], rho, rho_w, ni,deq_data['W_ast'] )

    # Calculate D_ast
    #deq_data['D_ast'] = calculate_D_ast(rho, rho_w, deq_data['d_eq_m'], w, mu)

    # Calculate settling velocity using Dietrich formula
    deq_data['v_diet'] = deq_data['W']/10
    return deq_data

def calculate_dioguardi_velocity(Re, Psi):
    term1 = (24 / Re) * ((1 - Psi) / Re + 1) ** 0.25
    term2 = (24 / Re) * (0.1806 * Re ** 0.6459) * Psi ** (-0.08 * Re)
    term3 = 0.4251 / (1 + 6880.95 / Re * Psi ** 5.05)
    
    C_D = term1 + term2 + term3
    return C_D

# Define the function to calculate settling velocity using Dioguardi formula
def calculate_v_s(d_eq, psi, ni, rho, rho_w, g):
    # Initial guess for settling velocity
    v_s = 0.01  # Adjust the initial guess
    
    # Maximum number of iterations
    max_iter = 1000
    
    for i in range(max_iter):
        # Calculate Reynolds number
        Re = d_eq * v_s / ni
        
        # Calculate drag coefficient using the Dioguardi formula
        CD = calculate_dioguardi_velocity(Re, psi)
        
        # Calculate new settling velocity using updated drag coefficient
        v_s_new = np.sqrt(4 / 3 * d_eq / CD * ((rho - rho_w) / rho_w) * g)
        
        # Calculate the residual sum of squares (RSS)
        RSS = (v_s - v_s_new) ** 2
        
        # Update the settling velocity for the next iteration
        v_s = v_s_new
        
        # Check if the RSS is less than the threshold
        if RSS < 1e-100:
            break
    else:
        print("Maximum number of iterations reached. Convergence not achieved.")
    
    return v_s

# Define an empty list to store settling velocities
dioguardi_settling_velocities = []

# Iterate over each row in the DataFrame
for index, row in D.iterrows():
    # Calculate shape factor Psi
    psi = row['Sphericity'] / row['Circularity']
    
    # Calculate settling velocity using Dioguardi formula for the current particle
    v_s_dioguardi = calculate_v_s(row['d_eq_m'], psi, ni, rho, rho_w, g)
    
    # Append the settling velocity to the list
    dioguardi_settling_velocities.append(v_s_dioguardi)

# Add Dioguardi settling velocities to the DataFrame
D['v_dioguardi'] = dioguardi_settling_velocities

def calculate_drag_coefficient_Swamee(d_eq, CSF, Re):
    term1 = 48.5 / (((1 + 4.5 * CSF ** 0.35) ** 0.8) * (Re ** 0.64))
    term2 = ((Re / (Re + 100 + 100 * CSF)) ** 0.32) * (1 / (CSF ** 18 + 1.05 * CSF ** 0.8))
    
    Cd = (term1 + term2) ** 1.25
    return Cd

def calculate_swamee_velocity(d_eq, CSF, psi, ni, rho, rho_w, g):
    # Calculate drag coefficient using Swamee's formula
    
    # Initial guess for settling velocity
    v_s = 0.01  # Adjust the initial guess
    
    # Maximum number of iterations
    max_iter = 1000
    
    for i in range(max_iter):
        # Calculate Reynolds number
        Re = d_eq * v_s / ni
        Cd = calculate_drag_coefficient_Swamee(d_eq, CSF, Re)
        # Calculate new settling velocity using updated drag coefficient
        v_s_new = np.sqrt(4 / 3 * d_eq / Cd * ((rho - rho_w) / rho_w) * g)
        
        # Calculate the residual sum of squares (RSS)
        RSS = (v_s - v_s_new) ** 2
        
        # Update the settling velocity for the next iteration
        v_s = v_s_new
        
        # Check if the RSS is less than the threshold
        if RSS < 1e-100:
            break
    else:
        print("Maximum number of iterations reached. Convergence not achieved.")
    
    return v_s

swamee_settling_velocities = []

# Iterate over each row in the DataFrame
for index, row in D.iterrows():
    # Calculate shape factor Psi
    
    # Calculate settling velocity using Dioguardi formula for the current particle
    v_swamee = calculate_swamee_velocity(row['d_eq_m'], row['CSF'], psi,ni, rho, rho_w, g)
    
    # Append the settling velocity to the list
    swamee_settling_velocities.append(v_swamee)

# Add Dioguardi settling velocities to the DataFrame
D['v_swamee'] = swamee_settling_velocities


def calculate_K1(sphericity):
    K1 = (1/3 + 2/3 * sphericity ** -0.5) ** -1
    return K1

def calculate_K2(sphericity):
    K2 = 10 ** (1.8148 * (-np.log10(sphericity)) ** 0.5743)
    return K2

def calculate_drag_coefficient_Ganser(Re, K1, K2):
    term1 = 24 / (K1 * K2 * Re)
    term2 = 1 + 0.118 * (K1 * K2 * Re) ** 0.6567
    term3 = 0.435 / (1 + 3305 / (K1 * K2 * Re))
    
    Cd = K2 * (term1 * term2 + term3)
    
    return Cd

# Function to calculate the settling velocity using the Ganser formula
def calculate_v_s_Ganser(d_eq, psi, ni, rho, rho_w, g, K1, K2):
    # Initial guess for settling velocity
    v_s = 0.01  # Adjust the initial guess
    
    # Maximum number of iterations
    max_iter = 1000
    
    for i in range(max_iter):
        # Calculate Reynolds number
        Re = d_eq * v_s / ni
        
        # Calculate drag coefficient using the Ganser formula
        CD = calculate_drag_coefficient_Ganser(Re, K1, K2)
        
        # Calculate new settling velocity using updated drag coefficient
        v_s_new = np.sqrt(4 / 3 * d_eq / CD * ((rho - rho_w) / rho_w) * g)
        
        # Calculate the residual sum of squares (RSS)
        RSS = (v_s - v_s_new) ** 2
        
        # Update the settling velocity for the next iteration
        v_s = v_s_new
        
        # Check if the RSS is less than the threshold
        if RSS < 1e-100:
            break
    else:
        print("Maximum number of iterations reached. Convergence not achieved.")
    
    return v_s

# Iterate over each row in the DataFrame to calculate settling velocity for each particle
def calculate_settling_velocities_Ganser(df, ni, rho, rho_w, g):
    settling_velocities2 = []
    
    for index, row in df.iterrows():
        # Calculate shape factor Psi
        psi = row['Sphericity'] / row['Circularity']
        
        # Calculate K1 and K2 based on particle properties
        K1 = calculate_K1(row['Sphericity'])
        K2 = calculate_K2(row['Sphericity'])
        
        # Calculate settling velocity using Ganser formula for the current particle
        v_s_Ganser = calculate_v_s_Ganser(row['d_eq_m'], psi, ni, rho, rho_w, g, K1, K2)
        
        # Append the settling velocity to the list
        settling_velocities2.append(v_s_Ganser)
    
    return settling_velocities2

# Calculate settling velocities using Ganser formula
settling_velocities2_Ganser = calculate_settling_velocities_Ganser(D, ni, rho, rho_w, g)

# Add Ganser settling velocities to the DataFrame
D['v_ganser'] = settling_velocities2_Ganser
# Parameters

P = 0.6  # dimensionless parameter
deq_file = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\particle_data_new.xlsx"

# Calculate settling velocity using Dietrich formula
settling_data = dietrich_settling_velocity(D['d_eq_m'], rho, ni, rho_w, deq_file, D['CSF'], P)

# Add Dietrich settling velocities to the DataFramerdf
D['v_dietrich'] = settling_data['v_diet']
D['v_stokes'] = settling_data['v_stk']
D['Re'] = settling_data['Re']
# Save the results to the same folder as the input Excel file
output_file = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\combined_settling_velocity_results.xlsx"
D.to_excel(output_file, index=False)

print("Data saved to Excel file:", output_file)
