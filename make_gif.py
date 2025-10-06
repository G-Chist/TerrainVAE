# CREDIT: https://github.com/Bhuvan-kio/GIF-Maker/blob/main/gif.py

#
# Creates an animated GIF from a sequence of PNG images.
#
# This script reads all .png files from a specified directory, sorts them in
# natural order to ensure the correct frame sequence, and then compiles them
# into a single looping GIF file.
#

# Import necessary libraries
import imageio.v3 as gif  # For reading images and writing the GIF
import os                 # For interacting with the file system (e.g., listing files)
import natsort            # For sorting filenames in natural order

# --- Configuration ---
# EDIT THESE VALUES
# Set the path to the directory containing the source PNG images.
img_dir = "results/"

# Set the path and filename for the final output GIF.
output = "results/progress.gif"

# Set the desired frames per second (FPS) for the animation.
fps = 10
# ---------------------

try:
    # Get a list of all files in the directory that end with .png
    filenames = [f for f in os.listdir(img_dir) if f.endswith('.png') and 'reconstruction' in f]
except FileNotFoundError:
    # Handle cases where the specified directory does not exist.
    print(f"Error: The folder '{img_dir}' was not found.")
    exit() # Stop the script if the folder is missing.

# Sort the images using natural sort order.
# This ensures that 'frame10.png' comes after 'frame9.png', not after 'frame1.png'.
sorted_images = natsort.natsorted(filenames)
print(f"Found and sorted {len(sorted_images)} images.")

# Create an empty list to hold the image data.
images = []

# Loop through each sorted filename and read the image data.
for filename in sorted_images:
    # Construct the full path to the image file.
    file_path = os.path.join(img_dir, filename)
    # Read the image file and append its data to the 'images' list.
    images.append(gif.imread(file_path))

print(f"Creating GIF '{output}' at {fps} FPS...")

# Write the collected images to a GIF file.
gif.imwrite(
    output,                  # The output file path.
    images,                  # The list of image frames.
    duration=(1000 / fps),   # Duration of each frame in milliseconds (1000ms / fps).
    loop=0                   # 0 means the GIF will loop indefinitely.
)

print("GIF created successfully!")
