# utils/debug_caption_loader.py
import pickle

captions = pickle.load(open('data/cub_text_embeddings/captions.pickle', 'rb'))
filenames = pickle.load(open('data/cub_text_embeddings/filenames.pickle', 'rb'))

print(f"Total images: {len(captions)}")
print("Example captions for first image:")
for i, cap in enumerate(captions[0]):
    print(f"{i+1}: {cap}")