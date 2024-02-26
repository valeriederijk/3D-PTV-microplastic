# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:57:52 2023

@author: valer
"""
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import csv
import os

class ImageCropper:
    def __init__(self, master, image_path):
        self.master = master
        self.master.title("Image Cropper")

        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Resize the image while maintaining its aspect ratio
        max_size = 800  # Adjust this value as needed
        height, width, _ = self.image_rgb.shape

        if max(height, width) > max_size:
            ratio = max_size / max(height, width)
            new_height = int(height * ratio)
            new_width = int(width * ratio)
            self.image_rgb = cv2.resize(self.image_rgb, (new_width, new_height))

        self.image_tk = ImageTk.PhotoImage(Image.fromarray(self.image_rgb))

        # Set the window size based on the screen resolution
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Set canvas size to match the image dimensions
        canvas_width = self.image_rgb.shape[1]
        canvas_height = self.image_rgb.shape[0]

        # Calculate the window size, ensuring it fits within the screen
        window_width = min(screen_width, canvas_width)
        window_height = min(screen_height, canvas_height)

        self.canvas = tk.Canvas(self.master, width=window_width, height=window_height)
        self.canvas.pack()

        self.rect = None
        self.start_x = None
        self.start_y = None
        self.crop_params = None  # Variable to store crop parameters

        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Add a button to save crop parameters
        self.save_button = tk.Button(self.master, text="Save Crop Parameters", command=self.save_crop_parameters)
        self.save_button.pack()


    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if self.rect:
            self.canvas.delete(self.rect)

        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

    # Calculate the scaling factor
        scale_x = self.image.shape[1] / self.canvas.winfo_width()

        scale_y = self.image.shape[0] / self.canvas.winfo_height()

    # Apply the scaling factor to the crop parameters
        self.crop_params = {
            'x': int(min(self.start_x * scale_x, end_x * scale_x)),
            'y': int(min(self.start_y * scale_y, end_y * scale_y)),
            'width': int(abs(end_x - self.start_x) * scale_x),
            'height': int(abs(end_y - self.start_y) * scale_y)
            }

        print("Crop Parameters:", self.crop_params)

    def save_crop_parameters(self):
        if self.crop_params:
            csv_file_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\crop_parameters_left.csv"
            with open(csv_file_path, 'w', newline='') as csv_file:
                fieldnames = ['x', 'y', 'width', 'height']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                # Write header
                writer.writeheader()

                # Write data
                writer.writerow(self.crop_params)

            print(f"Crop parameters saved to {csv_file_path}")
        else:
            print("No crop parameters to save.")

def main():
    root = tk.Tk()

    # Provide the path to your image file here
    image_path = r"C:\Users\valer\OneDrive - Wageningen University & Research\experiment_Alg4_221_pvc\output\frame_205.png"
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    cropper = ImageCropper(root, image_path)
    root.mainloop()

if __name__ == "__main__":
    main()
