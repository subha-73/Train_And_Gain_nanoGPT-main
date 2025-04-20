import pandas as pd

# Load the dataset
df = pd.read_csv('C:/Users/subha/Downloads/nanoGPT-master/nanoGPT-master/data/spotify_lyrics/spotify_millsongdata.csv')

# Extract the 'text' column (lyrics)
lyrics = df['text'].dropna()  # Remove any NaN values (if present)

# Save the lyrics to a text file
with open('C:/Users/subha/Downloads/nanoGPT-master/nanoGPT-master/data/spotify_lyrics/spotify_lyrics.txt', 'w', encoding='utf-8') as file:
    for lyric in lyrics:
        file.write(lyric.strip() + '\n')  # Ensure each lyric is on a new line
