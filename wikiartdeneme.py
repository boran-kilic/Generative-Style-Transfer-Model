import pandas as pd
from PIL import Image
import io
import glob
import os

parquet_files = glob.glob('wikiart/data/*.parquet')
filtered_dfs = []
total_files = len(parquet_files)
print(f"Processing {total_files} parquet files...")

processed_files = 0


for file in parquet_files:
    df_temp = pd.read_parquet(file)
    filtered_df = df_temp[df_temp['artist'] == 22]
    if not filtered_df.empty:
        filtered_dfs.append(filtered_df)
    processed_files += 1
    print(f"Processed {processed_files} out of {total_files} files...")

if filtered_dfs:
    final_df = pd.concat(filtered_dfs, ignore_index=True)
    final_df.to_parquet('wikiart/data/artist_22_dataset.parquet')
    print(f"Created dataset with {len(final_df)} images from artist ID 22")
else:
    print("No images found for artist ID 22")

# df = pd.read_parquet('wikiart/data/train-00000-of-00072.parquet')
# print(df.head())
# first_image_bytes = df.iloc[1]['image']['bytes']  # Assuming 'bytes' is the column name
# filtered_df = df[df['artist'] == 22]
# print(filtered_df.head())
# image = Image.open(io.BytesIO(first_image_bytes))
# image.show() 

