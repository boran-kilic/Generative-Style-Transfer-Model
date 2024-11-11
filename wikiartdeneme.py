import pandas as pd
from PIL import Image
import io
import glob
import os

# Get list of all parquet files in wikiart/data directory
parquet_files = glob.glob('wikiart/data/*.parquet')

# Create empty list to store filtered dataframes
filtered_dfs = []
# Get total number of parquet files for progress tracking
total_files = len(parquet_files)
print(f"Processing {total_files} parquet files...")

# Initialize counter for progress tracking
processed_files = 0


# Read and filter each parquet file
for file in parquet_files:
    df_temp = pd.read_parquet(file)
    # Filter for artist id 22
    filtered_df = df_temp[df_temp['artist'] == 22]
    if not filtered_df.empty:
        filtered_dfs.append(filtered_df)
    processed_files += 1
    print(f"Processed {processed_files} out of {total_files} files...")

# Combine all filtered dataframes
if filtered_dfs:
    final_df = pd.concat(filtered_dfs, ignore_index=True)
    # Save combined filtered data to new parquet file
    final_df.to_parquet('wikiart/data/artist_22_dataset.parquet')
    print(f"Created dataset with {len(final_df)} images from artist ID 22")
else:
    print("No images found for artist ID 22")

""" # Load the Parquet file
df = pd.read_parquet('wikiart/data/train-00000-of-00072.parquet')
print(df.head()) """
# Get the first image bytes
""" first_image_bytes = df.iloc[1]['image']['bytes']  # Assuming 'bytes' is the column name """
""" filtered_df = df[df['artist'] == 22]
print(filtered_df.head()) """
# Convert bytes to image
""" image = Image.open(io.BytesIO(first_image_bytes))

# Display the image
image.show()  # This will open the image in your default image viewer
 """