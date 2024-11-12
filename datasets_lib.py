from datasets import load_dataset

# Load the dataset from the local cache
dataset = load_dataset("huggan/wikiart", cache_dir="~/.cache/huggingface/datasets")

# Define a mapping from integer labels to artist names
artist_mapping = {
    22: "Vincent van Gogh",
    15: "Pablo Picasso",
    21: "Salvador Dali"
    # Add other mappings as needed
    # You may need to print all unique values in 'artist' to map all relevant labels
}

# Filter based on mapped names
target_artists = ["Salvador Dali", "Vincent van Gogh", "Pablo Picasso"]
target_artist_ids = [id for id, name in artist_mapping.items() if name in target_artists]

# Filter the dataset
filtered_dataset = dataset["train"].filter(lambda example: example['artist'] in target_artist_ids)

save_directory = "/Users/furkangurdogan/Desktop/filtered"
if len(filtered_dataset) > 0:
    filtered_dataset.save_to_disk(save_directory)
else:
    print("No matching records found for the specified artists.")

