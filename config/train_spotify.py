# train a miniature character-level model for Spotify lyrics
# good for debugging and playing with a small dataset

out_dir = 'out-spotify-lyrics-char'  # Output directory for this model
eval_interval = 250  # Keep frequent because we might overfit
eval_iters = 200
log_interval = 10  # Log interval for printing loss

# we expect to overfit on this small dataset, so only save when validation loss improves
always_save_checkpoint = False

wandb_log = False  # Set to True if you want to log to Weights & Biases
wandb_project = 'spotify-lyrics-char'  # Name of the wandb project
wandb_run_name = 'mini-gpt-spotify'

# Dataset name should correspond to the path where your data is stored
dataset = 'spotify_lyrics'  # The name of the dataset directory (inside data/spotify_data/spotify_lyrics)
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # Context of up to 256 previous characters

# Baby GPT model architecture
n_layer = 6  # Number of transformer layers
n_head = 6   # Number of attention heads
n_embd = 384  # Embedding dimension
dropout = 0.2  # Dropout rate

# Learning rate and optimization settings
learning_rate = 1e-3  # A bit higher learning rate for small models
max_iters = 5000
lr_decay_iters = 5000  # Make this equal to max_iters for consistency
min_lr = 1e-4  # Learning rate decay factor
beta2 = 0.99  # Larger beta2 due to small dataset size

warmup_iters = 100  # Number of warmup iterations for learning rate decay

# Uncomment these if you're working on CPU or need optimizations
# device = 'cpu'  # Use this if you're running on CPU only (not recommended for training on large datasets)
# compile = False  # Disable torch compiling if it's causing issues