# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:41:24 2023

@author: valer
"""
import cv2
import re
import os
from skimage import io, color, filters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


frames_dir =r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\frames_right"
frame_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\frames_right\frame_1810.png"
B = io.imread(frame_path)
K = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
# Invert the grayscale image
inverted_image = 255 - K

height, width, _ = B.shape
print ('{height}')

print("Height:", height)
print("Width:", width)
# Set the percentage of the image to keep (adjust as needed)
percentage_height_to_keep = 65
percentage_width_to_keep =70

# Calculate the pixels to keep based on the percentage
height, width = inverted_image.shape[:2]
crop_height = int(height * (percentage_height_to_keep / 100))
crop_width = int(width * (percentage_width_to_keep / 100))

# Crop the region of interest excluding the top and bottom edges
inverted_image_roi = inverted_image[int((height - crop_height) / 2):int((height + crop_height) / 2),
                                    int((width - crop_width) / 2):int((width + crop_width) / 2)]




# Display the original inverted image and the cropped region
plt.subplot(121), plt.imshow(inverted_image, cmap='gray'), plt.title('Original Inverted Image')
plt.subplot(122), plt.imshow(inverted_image_roi, cmap='gray'), plt.title('Cropped Region')
plt.show()
# Threshold the inverted grayscale image to segment the particle (assuming particle is white)
threshold_value = 120 # Adjust this threshold value as needed
_, binary_image = cv2.threshold(inverted_image_roi, threshold_value, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found:", len(contours))
# If there is at least one contour
if len(contours) > 0:
    # Assuming you want the largest contour (particle)
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)

    # Particle coordinates in the cropped region
    particle_x_roi = int(M['m10'] / M['m00'])
    particle_y_roi = int(M['m01'] / M['m00'])

    # Particle coordinates normalized for the original image
    particle_x_original = particle_x_roi + int((width - crop_width) / 2)
    particle_y_original = particle_y_roi + int((height - crop_height) / 2)

    print("Particle Coordinates (Original Image):")
    print("X:", particle_x_original)
    print("Y:", particle_y_original)

    # Display the original inverted image and the cropped region with the highlighted particle
    plt.subplot(121), plt.imshow(inverted_image, cmap='gray'), plt.title('Original Inverted Image')
    plt.scatter(particle_x_original, particle_y_original, color='red', marker='x', s=50)  # Highlight the particle
    plt.subplot(122), plt.imshow(inverted_image_roi, cmap='gray'), plt.title('Cropped Region')
    plt.show()
else:
    print("No particles detected in the frame.")