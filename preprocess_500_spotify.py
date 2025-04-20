import pandas as pd

# Load the dataset
df = pd.read_csv('C:/Users/SK/Downloads/nanoGPT-master/nanoGPT-master/data/spotify_data/spotify_millsongdata.csv')

# Extract only the text column (lyrics)
lyrics = df['text'].dropna().values

# Join all lyrics into a single string
full_text = "\n".join(lyrics)

# Optional: Clean text (remove unnecessary characters, special symbols, etc.)
# For example, you could remove excessive spaces or newline characters.
full_text = full_text.replace('\n\n', '\n')

# Write the cleaned text to a new file
with open('spotify_lyrics.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

print("Text preprocessing done.")
