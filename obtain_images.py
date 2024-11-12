from datasets import load_from_disk
from PIL import Image
import os

from datasets import load_from_disk
import os

# Path to the directory with Arrow files
dataset_path = "/Users/furkangurdogan/Desktop/filtered"
dataset = load_from_disk(dataset_path)

# Define the artist mapping
artist_mapping = {
    22: "Vincent van Gogh",
    15: "Pablo Picasso",
    21: "Salvador Dali"
    # Add other mappings as needed
    # You may need to print all unique values in 'artist' to map all relevant labels
}

# Directory to save extracted images
image_save_dir = "/Users/furkangurdogan/Desktop/extracted_images"
os.makedirs(image_save_dir, exist_ok=True)

# Iterate over the dataset and save each image
for idx, example in enumerate(dataset):
    image = example['image']  # Extract the image as a PIL image
    artist_label = example['artist']  # Get the artist label (integer)

    # Get the artist name from the mapping, or use 'unknown' if not found
    artist_name = artist_mapping.get(artist_label, 'unknown')
    filename = f"{artist_name}_{idx}.jpg"
    file_path = os.path.join(image_save_dir, filename)
    
    # Save the image
    image.save(file_path)