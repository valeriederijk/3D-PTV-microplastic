# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:48:02 2024

@author: valer
"""
#try different calibration

import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import csv
import random

# functions needed later

def save_calibration_metrics(metrics, save_folder='plots'):
    file_path = os.path.join(save_folder, 'calibration_metrics.npz')
    np.savez(file_path, **metrics)
    print(f"Calibration metrics saved to {file_path}")

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


def save_metrics_to_csv(metrics, csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        for metric_name, metric_value in metrics.items():
            writer.writerow([metric_name, metric_value])
    print(f"Metrics saved to {csv_filename}")


# calibration
def calibrate_camera(images_folder, camera_index, selected_frame_numbers, save_folder='plots'):
    imgpoints = []
    objpoints = []
#See openCV documentation
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#here you define the shape of your calibration image
    rows = 4
    columns = 7
    world_scaling = 1.9

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    frames_processed = 0

    for frame_number in selected_frame_numbers:
        frame_name = f'frame_{frame_number:04d}.png'
        im_path = os.path.join(images_folder, frame_name)

        if os.path.exists(im_path):
            print(f"Found calibration image: {im_path}")
            gray = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
            #ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
            
            #adjustments are here
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            #enhanced_image = cv.equalizeHist(gray)
            ret, corners = cv.findChessboardCorners(gray, (rows, columns), criteria, flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_NORMALIZE_IMAGE)
            #corners = cv.findChessboardCorners(gray, (rows, columns), None)
            if not ret:
                # Skip the frame if corners cannot be detected
                print(f"Could not find corners in image. Skipping: {frame_name}")
                continue

            print(f"Detected corners in image: {len(corners)}")

            #conv_size = (11, 11)
            #corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)

            # Save the image with a new filename based on the original frame number and camera index
            new_filename = f'calibration_frame_cam{camera_index}_{frame_number:04d}.png'
            new_filepath = os.path.join(save_folder, new_filename)
            cv.imwrite(new_filepath, cv.imread(im_path))
            print(f"Saved calibration frame: {new_filepath}")

            objpoints.append(objp)
            imgpoints.append(corners)
            frames_processed += 1

        else:
            print(f"Could not find calibration image: {im_path}")

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (gray.shape[1], gray.shape[0]), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    # Save calibration metrics to CSV
    calibration_metrics = {
        'RMSE': ret,
        'CameraMatrix': mtx,
        'DistortionCoeffs': dist,
        'Rs': rvecs,
        'Ts': tvecs
    }
    save_metrics_to_csv(calibration_metrics, csv_filename=os.path.join(
        save_folder, f'calibration_metrics_cam{camera_index}.csv'))
    return mtx, dist

def save_selected_pairs_to_csv(selected_frame_numbers, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Selected Frame Numbers'])
        for frame_number in selected_frame_numbers:
            csv_writer.writerow([frame_number])


def stereo_calibrate(mtx1, dist1, mtx2, dist2, num_pairs=50, save_folder='plots'):
    c1_images_names = glob.glob(os.path.join(
        save_folder, 'calibration_frame_cam1_*.png'))
    c2_images_names = glob.glob(os.path.join(
        save_folder, 'calibration_frame_cam2_*.png'))

    print("Camera 1 image names:", c1_images_names)
    print("Camera 2 image names:", c2_images_names)

    # Ensure at least one image for each camera is found
    if not c1_images_names or not c2_images_names:
        print("Error: No calibration images found for one or both cameras.")
        return None, None

    pairs = []
    available_frame_numbers = [int(os.path.splitext(os.path.basename(fname))[0].split('_')[-1]) for fname in c1_images_names]

    # Add these print statements to debug
    print("Available frame numbers:", available_frame_numbers)
    #this ensures random selection of calibration images
    selected_frame_numbers = set()
    while len(pairs) < num_pairs:
        # Randomly select a frame number
        frame_number = random.choice(available_frame_numbers)
        frame_number_str = f'{frame_number:04d}'

        im1 = os.path.join(
            save_folder, f'calibration_frame_cam1_{frame_number_str}.png')
        im2 = os.path.join(
            save_folder, f'calibration_frame_cam2_{frame_number_str}.png')

        print(f"Selected frame number: {frame_number_str}")
        print(f"File paths: {im1}, {im2}")

        if os.path.exists(im1) and os.path.exists(im2):
            pairs.append((im1, im2))
            selected_frame_numbers.add(frame_number)
        else:
            print(f"One or both images for frame {frame_number_str} do not exist.")
            
    save_selected_pairs_to_csv(selected_frame_numbers, csv_filename=os.path.join(save_folder, 'selected_calibration_pairs_trial.csv'))

    # Ensure at least one pair of matching frames is found
    if not pairs:
        print("Error: No matching calibration frames found for both cameras.")
        return None, None

    print("Selected pairs for stereo calibration:")
    for pair in pairs:
        print(pair)

    c1_images = [cv.imread(im1) for im1, _ in pairs]
    # Check if images are successfully read
    for i, image in enumerate(c1_images):
        if image is not None:
            print(f"Successfully read image {i + 1} for camera 1.")
        else:
            print(f"Error: Failed to read image {i + 1} for camera 1.")

    c2_images = [cv.imread(im2) for _, im2 in pairs]
    for i, image in enumerate(c2_images):
        if image is not None:
            print(f"Successfully read image {i + 1} for camera 2.")
        else:
            print(f"Error: Failed to read image {i + 1} for camera 2.")

    pair_index = 0
    selected_frame1 = c1_images[pair_index]
    selected_frame2 = c2_images[pair_index]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # dont forget to change
    rows = 4
    columns = 7
    world_scaling = 1.9

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(
            gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(
            gray2, (rows, columns), None)

        print("Chessboard corners found in image 1:", c_ret1)
        print("Chessboard corners found in image 2:", c_ret2)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(
                gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(
                gray2, corners2, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    # stereocalibration_flags = cv.CALIB_ZERO_DISPARITY
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (
                                                                     width, height), criteria=criteria,
                                                                 flags=stereocalibration_flags)

    if ret < 0:
        print(f"Error: Stereo calibration failed with return code {ret}")
        return None, None

    rmse = np.sqrt(ret / len(objpoints))
    print('RMSE:', rmse)
    print(ret)

    # Additional code to compute and save metrics
    metrics = {
        'RMSE': rmse,
        'mtx1': CM1,
        'dist1': dist1,
        'mtx2': CM2,
        'dist2': dist2,
        'R': R,
        'T': T
    }
    # Save stereo calibration metrics to CSV
    save_metrics_to_csv(metrics, csv_filename=os.path.join(
        save_folder, 'stereo_calibration_metrics_trial.csv'))

    if save_folder:
        plt.figure(figsize=(10, 10))
        ax = [plt.subplot(2, 2, i + 1) for i in range(4)]

        for a, frame in zip(ax, c1_images):
            a.imshow(frame[:, :, [2, 1, 0]])
            a.set_xticklabels([])
            a.set_yticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0)
        save_plot(plt, "stereo_calibration_plot2_trial.png", save_folder)
    else:
        plt.show()

    return R, T, c1_images, c2_images


def triangulate(mtx1, mtx2, R, T, frame1, frame2, save_folder='plots'):
    # Select a random pair of images from the stereo calibration
    pair_index = random.randint(0, len(c1_images) - 1)
    frame1 = c1_images[pair_index]
    frame2 = c2_images[pair_index]

    # Find checkerboard corners in both images
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    ret1, corners1 = cv.findChessboardCorners(gray1, (4, 7), None)

    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    ret2, corners2 = cv.findChessboardCorners(gray2, (4, 7), None)

    if not ret1 or not ret2:
        print("Error: Checkerboard corners not found in one or both images.")
        return

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
        print('Triangulated point: ')
        print(triangulated_point)
        return triangulated_point

    p3ds = []
    for uv1, uv2 in zip(corners1, corners2):
        _p3d = DLT(P1, P2, uv1[0], uv2[0])
        p3ds.append(_p3d)

    p3ds = np.array(p3ds)

    # Calculate reprojection error
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

    save_metrics_to_csv(metrics, csv_filename=os.path.join(
        save_folder, 'triangulation_metrics.csv'))

# Plot the original and reprojected points
    plt.figure()
    plt.scatter(corners1[:, 0, 0], corners1[:, 0, 1],
                c='blue', label='Original Points (Image 1)')
    plt.scatter(corners2[:, 0, 0], corners2[:, 0, 1],
                c='green', label='Original Points (Image 2)')
    plt.scatter([p[0] for p in reproj_points1], [p[1] for p in reproj_points1],
                c='red', marker='x', label='Reprojected Points (Image 1)')
    plt.scatter([p[0] for p in reproj_points2], [p[1] for p in reproj_points2],
                c='purple', marker='x', label='Reprojected Points (Image 2)')

    plt.title('Original vs Reprojected Points')
    plt.legend()

    if save_folder:
        save_plot(plt, "reprojection_plot_trial.png", save_folder)
    else:
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Automatically adjust axes limits
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(np.min(p3ds[:, 2]) - 5, np.max(p3ds[:, 2]) + 5)

    # Scatter plot
    ax.scatter(p3ds[:, 0], p3ds[:, 1], p3ds[:, 2], c='red', marker='o')

    # Set plot title
    ax.set_title('Triangulated 3D Points')

    # Show the interactive plot
    plt.show(block=True)

if __name__ == "__main__":
    
    # Code here will only run if the script is executed directly
# change range to what is applicable
    save_folder = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\calibration3"
    start_frame, end_frame = (1, 950)
    max_frames = 800
    selected_frame_numbers = random.sample(range(start_frame, end_frame + 1), max_frames)
    mtx1, dist1 = calibrate_camera(images_folder=r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\frames_left",
                                   camera_index=1, selected_frame_numbers = selected_frame_numbers, save_folder=save_folder)
    mtx2, dist2 = calibrate_camera(images_folder=r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\frames_right",
                               camera_index=2, selected_frame_numbers = selected_frame_numbers, save_folder=save_folder)

# After stereo calibration, you can use the following modification

    R, T, c1_images, c2_images = stereo_calibrate( mtx1, dist1, mtx2, dist2, 15, save_folder)

# Now you can call the triangulate function with the randomly selected pair of images
    triangulate(mtx1, mtx2, R, T, c1_images, c2_images, save_folder)
    
    calibration_metrics = {
            'mtx1': mtx1,   
            'dist1': dist1,
            'mtx2': mtx2,
            'dist2': dist2,
            'R': R,
            'T': T
        }
    save_calibration_metrics(calibration_metrics, save_folder)
    
    global_mtx1 = mtx1
    


