import os
import pickle
import numpy as np

# input file
input_file_path = os.path.join("data", "spotify_lyrics", "spotify_lyrics.txt")
with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # encoder: str -> list of ints
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: list of ints -> str

# encode the entire dataset and store it into a numpy array
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
os.makedirs(os.path.join("data", "spotify_lyrics"), exist_ok=True)
train_ids.tofile(os.path.join("data", "spotify_lyrics", "train.bin"))
val_ids.tofile(os.path.join("data", "spotify_lyrics", "val.bin"))

# save the meta info as well
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join("data", "spotify_lyrics", "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Done. Data is prepared.")