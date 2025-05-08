import os
import json
from pathlib import Path
from tqdm import tqdm

image_root = Path("./data/CUB_200_2011/CUB_200_2011/images")
caption_root = Path("./data/cub_text_embeddings/text")
output_path = "./data/cub_train.json"

data = []

# Iterate through each class folder
for class_folder in tqdm(sorted(image_root.glob("*"))):
    class_name = class_folder.name

    caption_class_folder = caption_root / class_name
    if not caption_class_folder.exists():
        print(f"[Warning] No captions for class: {class_name}")
        continue

    # For each image in that class
    for image_file in sorted(class_folder.glob("*.jpg")):
        image_name = image_file.stem  # without .jpg

        # Match caption by image_name prefix
        matching_captions = list(caption_class_folder.glob(f"{image_name}*.txt"))

        if not matching_captions:
            print(f"[Warning] Caption not found for image: {image_file.name}")
            continue

        # Load the first caption
        with open(matching_captions[0], "r") as f:
            caption = f.read().strip()

        data.append({
            "image": str(image_file),
            "caption": caption
        })

# Save to JSON
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Saved {len(data)} image-caption pairs to {output_path}")