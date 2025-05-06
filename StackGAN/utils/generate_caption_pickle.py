# utils/generate_caption_pickle.py
import os
import pickle

text_dir = 'data/cub_text_embeddings/text'
captions = []
filenames = []

for class_folder in sorted(os.listdir(text_dir)):
    class_path = os.path.join(text_dir, class_folder)
    if not os.path.isdir(class_path): continue

    for txt_file in sorted(os.listdir(class_path)):
        if not txt_file.endswith('.txt'): continue
        txt_path = os.path.join(class_path, txt_file)

        # Read and clean lines
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().lower() for line in lines if line.strip()]
            captions.append(lines[:5])  # Only take first 5
            filenames.append(txt_file.replace('.txt', ''))

print(f"Total image captions: {len(captions)}")

# Save
# pickle.dump(captions, open('data/cub_text_embeddings/captions.pickle', 'wb'))
pickle.dump(filenames, open('data/cub_text_embeddings/filenames.pickle', 'wb'))
