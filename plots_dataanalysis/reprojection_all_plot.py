# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:13:30 2024

@author: valer
"""
import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import glob
import os

def load_calibration_metrics(file_path):
    data = np.load(file_path)
    mtx1 = data['mtx1']
    dist1 = data['dist1']
    mtx2 = data['mtx2']
    dist2 = data['dist2']
    R = data['R']
    T = data['T']
    return mtx1, dist1, mtx2, dist2, R, T

def save_plot(figure, filename, save_folder='plots'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    figure_path = os.path.join(save_folder, filename)
    figure.savefig(figure_path)
    plt.close()


def reprojection_error(points_2d, projected_points_2d):
    points_2d = np.array(points_2d)
    projected_points_2d = np.array(projected_points_2d)
    return np.sqrt(np.mean(np.sum((points_2d - projected_points_2d)**2, axis=1)))

def load_calibration_images(camerafolder):
    # Define paths to calibration images
    c1_images_names = glob.glob(os.path.join(
        camerafolder, 'calibration_frame_cam1_*.png'))
    c2_images_names = glob.glob(os.path.join(
        camerafolder, 'calibration_frame_cam2_*.png'))

    print("Camera 1 image names:", c1_images_names)
    print("Camera 2 image names:", c2_images_names)

    # Ensure at least one image for each camera is found
    if not c1_images_names or not c2_images_names:
        print("Error: No calibration images found for one or both cameras.")
        # You might want to handle this error case appropriately
        # Returning None, None for now
        return None, None
    else:
        # Load the images using OpenCV
        c1_images = [cv.imread(image_path) for image_path in c1_images_names]
        c2_images = [cv.imread(image_path) for image_path in c2_images_names]
        return c1_images, c2_images


def triangulate(mtx1, dist1, mtx2, dist2, R, T, camerafolder, save_folder='plots'):
    c1_images, c2_images = load_calibration_images(camerafolder)
    if c1_images is None or c2_images is None:
        return
    
    found_corners = False
    while not found_corners:
        pair_index = random.randint(0, len(c1_images) - 1)
        frame1 = c1_images[pair_index]
        frame2 = c2_images[pair_index]

        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        ret1, corners1 = cv.findChessboardCorners(gray1, (4, 7), None)

        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        ret2, corners2 = cv.findChessboardCorners(gray2, (4, 7), None)

        if ret1 and ret2:
            found_corners = True

    corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria=(
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria=(
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1

    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2

    def DLT(P1, P2, point1, point2):
        A = [point1[1]*P1[2, :] - P1[1, :],
             P1[0, :] - point1[0]*P1[2, :],
             point2[1]*P2[2, :] - P2[1, :],
             P2[0, :] - point2[0]*P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))

        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices=False)

        triangulated_point = Vh[3, 0:3]/Vh[3, 3]
        return triangulated_point

    p3ds = []
    for uv1, uv2 in zip(corners1, corners2):
        _p3d = DLT(P1, P2, uv1[0], uv2[0])
        p3ds.append(_p3d)

    p3ds = np.array(p3ds)

    reproj_errors = []
    reproj_points1 = []
    reproj_points2 = []

    for uv1, uv2, p3d in zip(corners1, corners2, p3ds):
        reproj_point1, _ = cv.projectPoints(
            np.array([p3d]), RT1[:, :3], RT1[:, 3:], mtx1, dist1)
        reproj_point2, _ = cv.projectPoints(
            np.array([p3d]), RT2[:, :3], RT2[:, 3:], mtx2, dist2)

        reproj_errors.append(reprojection_error([uv1[0]], [
                             reproj_point1[0][0]]) + reprojection_error([uv2[0]], [reproj_point2[0][0]]))
        reproj_points1.append(reproj_point1[0][0])
        reproj_points2.append(reproj_point2[0][0])

    avg_reproj_error = np.mean(reproj_errors)
    rmse = np.sqrt(avg_reproj_error)
    print(f'Average Reprojection Error: {avg_reproj_error}')
    print(f'RMSE: {rmse}')

    metrics = {
        'AverageReprojectionError': avg_reproj_error,
        'RMSE': rmse
    }

    return avg_reproj_error, corners1, corners2, reproj_points1, reproj_points2

# Define paths to calibration metrics files and camera folders for each experiment
experiments = {
    "DW-1": {
        "calibration_metrics_file_path": r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\calibration\calibration_metrics.npz",
        "camerafolder": r"C:\Users\valer\OneDrive\Documents\master year 1\master wageningen\thesis\preliminaryscripts\experiment_22_12_pvc\calibration"
    },
    "DW-2": {
        "calibration_metrics_file_path": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\calibration\calibration_metrics.npz",
        "camerafolder": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_2112_pvc\calibration"
    },
    "Alg-1": {
        "calibration_metrics_file_path": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\calibration\calibration_metrics.npz",
        "camerafolder": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg1_191_pvc\calibration"
    },
    
    "Alg-2": {
        "calibration_metrics_file_path": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\calibration2\calibration_metrics.npz",
        "camerafolder": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg2_191_pvc\calibration2"
    },
    
    "Alg-3": {
        "calibration_metrics_file_path": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\calibration\calibration_metrics.npz",
        "camerafolder": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg3_191_pvc\calibration"
    },
    "Alg-4": {
        "calibration_metrics_file_path": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\calibration2\calibration_metrics.npz",
        "camerafolder": r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\calibration2"
    },
}

# Loop through experiments
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

for i, (experiment_name, experiment_data) in enumerate(experiments.items()):
    print(f"Processing {experiment_name}")
    calibration_metrics_file_path = experiment_data["calibration_metrics_file_path"]
    camerafolder = experiment_data["camerafolder"]
    mtx1, dist1, mtx2, dist2, R, T = load_calibration_metrics(calibration_metrics_file_path)
    if mtx1 is None or dist1 is None or mtx2 is None or dist2 is None or R is None or T is None:
        print(f"Skipping {experiment_name}: Failed to load calibration metrics.")
        continue
    avg_reproj_error, corners1, corners2, reproj_points1, reproj_points2 = triangulate(mtx1, dist1, mtx2, dist2, R, T, camerafolder)
    if avg_reproj_error is not None:
        row = i // 2
        col = i % 2
        print(f"{experiment_name} processed.")
        
        # Plot original and reprojected points
        axs[row, col].scatter(corners1[:, 0, 0], corners1[:, 0, 1],
                              c='blue', label='Original Points (Image 1)')
        axs[row, col].scatter(corners2[:, 0, 0], corners2[:, 0, 1],
                              c='green', label='Original Points (Image 2)')
        axs[row, col].scatter([p[0] for p in reproj_points1], [p[1] for p in reproj_points1],
                              c='red', marker='x', label='Reprojected Points (Image 1)')
        axs[row, col].scatter([p[0] for p in reproj_points2], [p[1] for p in reproj_points2],
                              c='purple', marker='x', label='Reprojected Points (Image 2)')
        axs[row, col].set_title(f'{experiment_name}')
        #axs[row, col].legend()
        
plt.tight_layout()
plt.show()
plt.show()