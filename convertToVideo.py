import cv2
import os

# Define the path where the images are stored
image_folder = 'frames'

# Define the output video file name and format
video_name = 'test3.mp4'

# Get a list of all image files in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
images.sort()  # Ensure the images are sorted in the correct order

# Read the first image to get the width and height
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

# Iterate through all images and write them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the VideoWriter object
video.release()

print("Video created successfully.")
