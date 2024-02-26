# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:21:34 2024

@author: valer
"""

import os
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from mpl_toolkits.mplot3d import Axes3D
import csv
import plotly.graph_objs as go
from plotly.subplots import make_subplots
def load_calibration_metrics(file_path):
    #define the file path location in the main function
    data = np.load(file_path)
    
    # Extract individual arrays from the dictionary
    mtx1 = data['mtx1']
    dist1 = data['dist1']
    mtx2 = data['mtx2']
    dist2 = data['dist2']
    R = data['R']
    T = data['T']

    return mtx1, dist1, mtx2, dist2, R, T

def calculate_speed(coord1, coord2, time_interval, pixels_per_cm):
    # Calculate speed based on the distance between coordinates and time interval
    distance = math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2) / pixels_per_cm
    speed = distance / time_interval
    return speed

def write_to_csv(output_file, particle_list):
    #computes average of calculated 2d speeds and saves to a csv file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Particle Index', 'Average Speed (m/s)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()

        for particle in particle_list:
            particle_index = particle['particle_index']
            average_speed = (sum(particle['speed']) / len(particle['speed']))/100 if len(particle['speed']) > 0 else 0.0
            writer.writerow({'Particle Index': particle_index, 'Average Speed (m/s)': average_speed})

def find_particle_coordinates(image, crop_height_percentage=100, crop_width_percentage=100, threshold_value=85):
    # main particle detection algorithm (used in analyze_frames function)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert the grayscale image
    inverted_image = 255 - gray_image

    # Set the percentage of the image height and width to keep (adjust as needed)
    percentage_height_to_keep = crop_height_percentage
    percentage_width_to_keep = crop_width_percentage
    threshold_value_cam = threshold_value
    # Calculate the pixels to keep based on the percentage, the percentages are defined in different script
    height, width = inverted_image.shape[:2]
    crop_height = int(height * (percentage_height_to_keep / 100))
    crop_width = int(width * (percentage_width_to_keep / 100))

    # Crop the region of interest excluding the top and bottom edges
    inverted_image_roi = inverted_image[int((height - crop_height) / 2):int((height + crop_height) / 2),
                                        int((width - crop_width) / 2):int((width + crop_width) / 2)]
    blurred_image = cv2.GaussianBlur(inverted_image_roi, (5, 5), 0)

    # Threshold the cropped image to segment the particle
    _, binary_image = cv2.threshold(blurred_image, threshold_value_cam, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 0  # Adjust this parameter as needed
    max_contour_area = 5000  # Adjust this parameter as needed
    contours = [cnt for cnt in contours if max_contour_area > cv2.contourArea(cnt) > min_contour_area]
    print("Number of contours found:", len(contours))
    
    #empty list to save coordinates
    particle_coordinates = []

    if len(contours) > 0:
        # Assuming you want the largest contour (particle)
        largest_contour = min(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M['m00'] != 0:
            # Particle coordinates in the cropped region
            particle_x_roi = int(M['m10'] / M['m00'])
            particle_y_roi = int(M['m01'] / M['m00'])

            # Particle coordinates normalized for the original image
            particle_x_original = particle_x_roi + int((width - crop_width) / 2)
            particle_y_original = particle_y_roi + int((height - crop_height) / 2)
            particle_coordinates.append((particle_x_original, particle_y_original))

    return particle_coordinates


def stereo_triangulation(pts1, pts2, mtx1, dist1, mtx2, dist2, R, T):
    #computes 3d coordinates from the position of camera 1; uses calibration matrix as defined
    pts1 = np.array(pts1).reshape(-1, 1, 2)
    pts2 = np.array(pts2).reshape(-1, 1, 2)
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    pts1_undistorted = cv2.undistortPoints(pts1, mtx1, dist1)
    pts2_undistorted = cv2.undistortPoints(pts2, mtx2, dist2)
    # Create 3x4 projection matrices
    projMatr1 = np.hstack((R, T))
    projMatr2 = np.hstack((np.eye(3), np.zeros((3, 1))))

    # Triangulate points
    points_3d_homogeneous = cv2.triangulatePoints(projMatr1, projMatr2, pts1_undistorted, pts2_undistorted)

    # Convert homogeneous coordinates to Cartesian coordinates
    points_3d_cartesian = cv2.convertPointsFromHomogeneous(points_3d_homogeneous.T.reshape(-1, 1, 4))

    return points_3d_cartesian.squeeze()

def detect_particles_in_frame(frame_path, detection_function, crop_height_percentage, crop_width_percentage, threshold_value):
    # allows you to use different detection algorithms; empty for now
    image = io.imread(frame_path)
    
    # Call the detection function with the specified crop_height_percentage and threshold_value
    return detection_function(image, crop_height_percentage, crop_width_percentage, threshold_value)

def analyze_frames(frames_dir_cam1, frames_dir_cam2, detection_function, starting_frame,
                   crop_height_percentage_cam1, crop_height_percentage_cam2,
                   crop_width_percentage_cam1, crop_width_percentage_cam2,
                   distance_threshold, pixels_per_cm, time_interval, output_csv_path,
                   max_frames_to_analyze, threshold_value_cam1,
                   threshold_value_cam2):
    #define empty list to save
    particle_list = []
    particle_coordinates_cam1 = []
    particle_coordinates_cam2 = [] 
    speed = []
    frames_analyzed = 0
    current_particle = None
    frames_without_particles = 0

    # Extract frame numbers from file names in cam1 folder
    frame_numbers_cam1 = sorted([int(filename.split('_')[-1].split('.')[0]) for filename in os.listdir(frames_dir_cam1)])

    for frame_number in frame_numbers_cam1[starting_frame:]:
        if frames_analyzed >= max_frames_to_analyze:
            break

        print(f"Analyzing frame: {frame_number}")
        #depends on the name of the files
        frame_path_cam1 = os.path.join(frames_dir_cam1, f"frame_{frame_number:04d}.png")
        frame_path_cam2 = os.path.join(frames_dir_cam2, f"frame_{frame_number:04d}.png")

        # Find particle coordinates in the current frame for both cameras
        particles_cam1 = detect_particles_in_frame(frame_path_cam1, detection_function, crop_height_percentage_cam1, crop_width_percentage_cam1, threshold_value_cam1)
        particles_cam2 = detect_particles_in_frame(frame_path_cam2, detection_function, crop_height_percentage_cam2, crop_width_percentage_cam2, threshold_value_cam2)

        # If particles were detected in the current frame
        if particles_cam1 and particles_cam2:
            frames_without_particles = 0  # Reset the counter
            if current_particle is None:
                current_particle = {
                    'particle_index': len(particle_list) + 1,
                    'frame_indices': [frame_number],
                    'coordinates_cam1': [particles_cam1[0]],
                    'coordinates_cam2': [particles_cam2[0]],
                    'speed': []  # Initialize the 'speed' key
                }
                particle_list.append(current_particle)
            else:
                # Check the distance between particles in two subsequent frames
                prev_particle_cam1 = current_particle['coordinates_cam1'][-1]
                prev_particle_cam2 = current_particle['coordinates_cam2'][-1]
                current_particle_cam1 = particles_cam1[0]
                current_particle_cam2 = particles_cam2[0]

                distance_cam1 = math.sqrt((current_particle_cam1[0] - prev_particle_cam1[0])**2 + (current_particle_cam1[1] - prev_particle_cam1[1])**2)
                distance_cam2 = math.sqrt((current_particle_cam2[0] - prev_particle_cam2[0])**2 + (current_particle_cam2[1] - prev_particle_cam2[1])**2)
                if distance_cam1 < distance_threshold and distance_cam2 < distance_threshold:
                    current_particle['frame_indices'].append(frame_number)
                    current_particle['coordinates_cam1'].append(current_particle_cam1)
                    current_particle['coordinates_cam2'].append(current_particle_cam2)
            
            # Calculate speed taking into account time interval
                if len(current_particle['frame_indices']) >= 2:
                    time_interval_between_frames = (frame_number - current_particle['frame_indices'][-2]) * 1/30
                    current_speed = calculate_speed(prev_particle_cam1, current_particle_cam1, time_interval_between_frames, pixels_per_cm)
                    
                    
                    if current_speed < 10:
                        current_particle['speed'].append(current_speed)
                        
                else:
                        current_particle = None  # Reset the current particle index

        else:
            frames_without_particles += 1
            if frames_without_particles > 5:
                current_particle = None  # Reset the current particle index

        frames_analyzed += 1


    write_to_csv(output_csv_path, particle_list)

    return particle_list, frames_dir_cam1, frames_dir_cam2, particle_coordinates_cam1, particle_coordinates_cam2


def triangulate_and_store_data(particle_list, frames_dir_cam1, frames_dir_cam2, detection_function, calibration_metrics_file, output_paths):
    # Load calibration metrics from the file
    # Load calibration metrics from the file
    mtx1, dist1, mtx2, dist2, R, T = load_calibration_metrics(calibration_metrics_file)

    # Load calibration metrics from the file
    mtx1, dist1, mtx2, dist2, R, T = load_calibration_metrics(calibration_metrics_file)

    columns = ['Particle', 'Frame', 'X', 'Y', 'Z']
    particle_data = []
    trajectories = {}
    speeds = {}

    # Particle list structure: [{'particle_index': 0, 'frame_indices': [1, 2, 3], 'coordinates_cam1': [(x1, y1), (x2, y2), ...], 'coordinates_cam2': [(x1, y1), (x2, y2), ...]}, ...]
    particle_list = particle_list

    for particle_info in particle_list:
        particle_index = particle_info['particle_index']
        frame_indices = particle_info['frame_indices']
        coordinates_cam1 = particle_info['coordinates_cam1']
        coordinates_cam2 = particle_info['coordinates_cam2']

        particle_trajectory = []  # Trajectory for the current particle
        particle_speeds = []  # Speeds for the current particle

        for i, (frame_index, (coord_cam1, coord_cam2)) in enumerate(zip(frame_indices, zip(coordinates_cam1, coordinates_cam2))):
            # Triangulate particle coordinates
            triangulated_coordinates = stereo_triangulation(coord_cam1, coord_cam2, mtx1, dist1, mtx2, dist2, R, T)
            # Calculate time difference dynamically
            delta_frame = frame_index - frame_indices[i - 1] if i > 0 else 0
            time_interval = 1/30
            delta_time = delta_frame * time_interval

            # Append data to particle_data
            particle_data.append({
                'Particle': particle_index,
                'Frame': frame_index,
                'X': triangulated_coordinates[0],
                'Y': triangulated_coordinates[1],
                'Z': triangulated_coordinates[2]
            })

            # Append data to particle_trajectory
            particle_trajectory.append({
                'Particle': particle_index,
                'Frame': frame_index,
                'X': triangulated_coordinates[0],
                'Y': triangulated_coordinates[1],
                'Z': triangulated_coordinates[2]
            })

            # Calculate speeds
            if i > 0:
                speed = np.linalg.norm(triangulated_coordinates - np.array([particle_trajectory[i-1]['X'], particle_trajectory[i-1]['Y'], particle_trajectory[i-1]['Z']])) / delta_time
                speed = speed
                particle_speeds.append(speed)

        # Update trajectories and speeds dictionaries
        trajectories[particle_index] = pd.DataFrame(particle_trajectory, columns=columns)
        speeds[particle_index] = particle_speeds

    # Create DataFrames from particle_data, trajectories, and speeds
    df_trajectories = pd.concat([trajectories[key] for key in trajectories])

   # Save x and y coordinates for all particles of the two cameras to CSV
    df_cam1 = pd.concat([df_trajectories[df_trajectories['Particle'] == particle_info['particle_index']][['Frame', 'X', 'Y']] for particle_info in particle_list])
    df_cam1.to_csv(output_paths['cam1_coordinates'], index=False)

    df_cam2 = pd.concat([df_trajectories[df_trajectories['Particle'] == particle_info['particle_index']][['Frame', 'X', 'Y']] for particle_info in particle_list])
    df_cam2.to_csv(output_paths['cam2_coordinates'], index=False)

    df_all_coordinates = pd.DataFrame(particle_data, columns=['Particle', 'Frame', 'X', 'Y', 'Z'])
    df_all_coordinates.to_csv(output_paths['3d_coordinates'], index=False)

    df_particle_data = pd.DataFrame(particle_data, columns=columns)

    return df_particle_data, df_trajectories, speeds

def calculate_and_plot_velocities(df_trajectories, speeds, output_paths_plots):
    # Calculate average velocity for each particle
    average_velocities = {}
    for particle_index, speed_list in speeds.items():
        if len(speed_list) > 0:
            average_velocity = np.mean(speed_list) / 100  # convert from cm/s to m/s
            average_velocities[particle_index] = average_velocity

    # Output average velocities to a CSV file
    average_velocities_df = pd.DataFrame(list(average_velocities.items()), columns=['Particle', 'Average Velocity'])
    average_velocities_df.to_csv(output_paths_plots['average_velocities'], index=False)
    
        # Output all speeds to a separate CSV file
    speeds_df = pd.DataFrame([(particle_index, speed) for particle_index, speed_list in speeds.items() for speed in speed_list], columns=['Particle', 'Speed'])
    speeds_df.to_csv(output_paths_plots['speeds'], index=False)

    # Plotting 3D trajectories
    plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')

    # Generate a colormap with unique colors for each particle
    colormap = plt.cm.rainbow(np.linspace(0, 1, len(df_trajectories['Particle'].unique())))
    min_x = df_trajectories['Y'].min()
    min_y = df_trajectories['Z'].min()

    for particle_index, data in df_trajectories.groupby('Particle'):
        x_coords = data['Y'].values - min_x  # Normalize x coordinates
        y_coords = data['Z'].values - min_y  # Normalize y coordinates
        z_coords = data['X'].values
        z_coords -= np.min(z_coords)
        # Plotting 3D trajectories with unique colors for each particle
        color = colormap[particle_index % len(colormap)]
        ax.plot(x_coords, y_coords, z_coords, label=f'Particle {particle_index}', color=color, linewidth=2)
    ax.set_xlabel('X Coordinate [cm]')
    ax.set_ylabel('Y Coordinate [cm]')
    ax.set_zlabel('Z Coordinate [cm]')
    ax.set_title('Particle 3D Trajectories')
    ax.set_box_aspect([np.ptp(y_coords), np.ptp(z_coords), np.ptp(x_coords)])
    #ax.set_zlim(0, np.max(df_trajectories['X']))
    #ax.legend()
    
    plt.savefig(output_paths_plots['3d_plot'])

    # Show the interactive plot
    
    
    plt.show(block=True)    
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# Generate a colormap with unique colors for each particle
    colormap = plt.cm.rainbow(np.linspace(0, 1, len(df_trajectories['Particle'].unique())))

# Initialize variables to store overall minimum coordinates# Find the minimum x and y coordinates to use as offsets
    min_x = df_trajectories['Y'].min()
    min_y = df_trajectories['Z'].min()
    
    coordinates_df = pd.DataFrame(columns=['Particle', 'X', 'Y', 'Z'])

    for particle_index, data in df_trajectories.groupby('Particle'):
        x_coords = data['Y'].values - min_x  # Normalize x coordinates
        y_coords = data['Z'].values - min_y  # Normalize y coordinates
        z_coords = data['X'].values
        z_coords -= np.min(z_coords) 
        
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
    coordinates_df.to_csv(output_paths_plots['3d_trajectories'], index=False)

    # Set layout
    fig.update_layout(scene=dict(
        xaxis=dict(range=[0, max(df_trajectories['Y'] - min_x)]),  # Set the x-axis limits
        yaxis=dict(range=[0, max(df_trajectories['Z'] - min_y)]),  # Set the y-axis limits
        #xaxis=dict(range=[0,10]),
       #yaxis=dict(range=[0,10]),
        xaxis_title='X coordinates [cm]',
        yaxis_title='Y coordinates [cm]',
        zaxis_title='Z Coordinate [cm]',  # Set aspectmode to 'manual' for custom aspect ratios  # Adjust ambient lighting for better visibility
    ), title='Particle 3D Trajectories')
        # Save the interactive plot
    fig.write_html(output_paths_plots['interactivefigure'])
        
    # Show the interactive plot
    fig.show()

    #####
    # Plotting speeds
    plt.figure(figsize=(12, 8))

    for particle_index, speed_list in speeds.items():
        plt.plot([speed / 100 for speed in speed_list], label=f'Particle {particle_index}')

    plt.xlabel('Frame')
    plt.ylabel('Speed (m/s)')  # Update ylabel to reflect the unit change
    plt.title('Particle Speeds')
    plt.legend()

    # Save speeds plot
    plt.savefig(output_paths_plots['speeds_plot'])
    plt.close()  # Close the figure to prevent it from being displayed

def main(starting_frame=0):
    # Path to camera frames
    frames_dir_cam1 = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\frames_left"
    frames_dir_cam2 = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\frames_right"
    
    #Path to calibration metrics
    calibration_metrics_file_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\calibration2\calibration_metrics.npz"
    
        # Output paths
    output_paths = {
          'cam1_coordinates': r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\cam1_coordinates.csv',
          'cam2_coordinates': r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\cam2_coordinates.csv',
          '3d_coordinates': r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\3dcoordinates.csv',
      }
    
    output_paths_plots = {
       'average_velocities': r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsexp.csv',
       'speeds': r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsexp_speeds.csv',
       '3d_plot':r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\3d_trajectories.png',
       '3d_trajectories': r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\3d_trajectories_trial.csv',
       'speeds_plot': r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\speeds_plottrial3.png',
       'interactivefigure': r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\3d_trajectories_trial.html',
   }


    # Define the detection function (you can replace this with your custom particle detection logic)
    detection_function = find_particle_coordinates

    # Analyze frames starting from the specified frame, fill in parameters here that are determined in the other scripts and are finetuned
    #specifically, change pixels_per_cm (based on initial camera crop and domain) and time_interal to match the inverse of fps
    #max_frames_to_analyze determines the amount of frames analyzed
    particle_list, frames_dir_cam1, frames_dir_cam2, particle_coordinates_cam1, particle_coordinates_cam2 = analyze_frames(frames_dir_cam1, frames_dir_cam2, detection_function, starting_frame,
               crop_height_percentage_cam1=70, crop_height_percentage_cam2=30,
               crop_width_percentage_cam1=70, crop_width_percentage_cam2=31,
               distance_threshold=55, pixels_per_cm=1554/30, time_interval=1/30,
               output_csv_path=r'C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\resultsspeed2d.csv',
               max_frames_to_analyze=21000, threshold_value_cam1= 135,
        threshold_value_cam2= 104)
    #Alg 1 --> cam 1 55, 65 cam 2 55, 65 threshold 1 = 130, threshold 2 = 78
    #Alg2 --> cam 1 55,65, cam 2 90, 75 , threshold 1 = 130, threshold 2 = 100
    #alg3 --> cam 1 70,65 cam 2 55, 60, threshold 1 = 125, threshold 2 = 110
    #alg4 --> cam 1 70, 50 cam 2 80, 55,  threshold 1 = 135, threshold 2 = 115
    # Triangulation done based on the stereo triangulation function
    particle_data, trajectories, speeds = triangulate_and_store_data(
        particle_list, frames_dir_cam1, frames_dir_cam2, detection_function, calibration_metrics_file_path, output_paths
    )
    
    #save_path_3d_2 = r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\3d_trajectories_trial.html"
    
    #control the output plots
    calculate_and_plot_velocities(trajectories, speeds, output_paths_plots)
    
#if you want to call parameters to other scripts this ensures that the entire script does not run again
if __name__ == "__main__":
    starting_frame = 1200  # You can set the starting frame value here; saves time
    main(starting_frame)
