# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:28:21 2024

@author: valer
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#circularity
def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        circularity = 0  # Handle the case where perimeter is zero
    else:
        circularity = (np.pi * (4 * area) ** 0.5) / perimeter
    return circularity
#sphericity

def calculate_sphericity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        sphericity = 0  # Handle the case where perimeter is zero
    else:
        sphericity = (4 * np.pi * area) / (perimeter ** 2)
    return sphericity
# roundness
def calculate_roundness(contour):
    area = cv2.contourArea(contour)
    if len(contour) < 5:
        print("Skipping contour: Insufficient points to fit an ellipse.")
        return None, None
    _, (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
    if minor_axis > major_axis:
        major_axis, minor_axis = minor_axis, major_axis  # Swap major and minor axes
    roundness = (4 * area) / (np.pi * (major_axis ** 2))
    aspect_ratio = major_axis / minor_axis if minor_axis != 0 else float('inf')  # handle division by zero
    return roundness, aspect_ratio
#corey shape factor
def calculate_csf(contour):
    if len(contour) < 5:
        print("Skipping contour: Insufficient points to fit an ellipse.")
        return None
    _, (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
    a = max(major_axis, minor_axis)
    b = min(major_axis, minor_axis)
    c = cv2.minEnclosingCircle(contour)[1] / 2
    #c = min(major_axis, minor_axis)  # Diameter to radius
    csf = c / np.sqrt(a * b)
    return csf

def calculate_equivalent_diameter(contour):
    area = cv2.contourArea(contour)
    equivalent_diameter = equivalent_diameter_cm = 2 * np.sqrt(area / np.pi) / pixels_per_cm
    return equivalent_diameter


# Load image
image_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\particles_Alg1_19_1.JPG"
image = cv2.imread(image_path)
#use height on ruler to defind this 
picture_height_cm = 8
pixels_per_cm = image.shape[0] / picture_height_cm
# Convert the image to grayscale for thresholding
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to segment the black particles on a white background
_, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Invert the thresholded image to make black particles white
thresholded = cv2.bitwise_not(thresholded)

# Find contours of particles
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours: {len(contours)}")

# Create empty lists to store results
particle_data = []

# Loop over each contour
for idx, contour in enumerate(contours):
    # Calculate sphericity
    sphericity = calculate_sphericity(contour)
    
    # Calculate circularity
    circularity = calculate_circularity(contour)
    
    roundness, aspect_ratio = calculate_roundness(contour)
    
    csf = calculate_csf(contour)
    deq = calculate_equivalent_diameter(contour)
    
    # Append results to the list
    particle_data.append({
        'Particle': idx + 1,
        'Sphericity': sphericity,
        'Circularity': circularity,
        'Roundness': roundness,
        'Aspect Ratio': aspect_ratio,
        'CSF': csf,
        'Equivalent Diameter': deq
    })

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(particle_data)

# Save the DataFrame to an Excel file
excel_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\particle_data_new.xlsx"
df.to_excel(excel_path, index=False)

print("Data saved to Excel file:", excel_path)

