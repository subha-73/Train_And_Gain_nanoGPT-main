"""
Sample from an untrained model using dynamic dataset directory.
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import argparse

# -----------------------------------------------------------------------------
# Argument parsing for dynamic dataset directory
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on: cpu or cuda')
parser.add_argument('--out_dir', type=str, default='out-untrained', help='Directory to save outputs')
parser.add_argument('--start', type=str, default="Once upon a time", help='Starting text prompt')
parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
parser.add_argument('--max_new_tokens', type=int, default=500, help='Max number of new tokens to generate')
parser.add_argument('--temperature', type=float, default=0.8, help='Randomness level for text generation')
parser.add_argument('--top_k', type=int, default=200, help='Restrict to top_k tokens at each step')
parser.add_argument('--seed', type=int, default=1337, help='Random seed')
parser.add_argument('--data_dir', type=str, default='data/spotify_lyrics', help='Directory of the dataset')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model with PyTorch 2.0')
args = parser.parse_args()

# ----------------------------------------------------------------------------- 

# Set random seed and backend settings
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in args.device else 'cpu'
dtype = 'float32'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load tokenizer metadata
meta_path = os.path.join(args.data_dir, 'meta.pkl')  # Use dynamic data_dir
load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = len(stoi)
    encode = lambda s: [stoi.get(c, stoi[' ']) for c in s]  # use space if char not found
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, falling back to GPT-2 encoding...")
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = 50257
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

    decode = lambda l: enc.decode(l)

# Create a new model config
gptconf = GPTConfig(
    block_size=256,
    vocab_size=vocab_size,
    n_layer=6,
    n_head=6,
    n_embd=384
)
model = GPT(gptconf)

model.eval()
model.to(args.device)
if args.compile:
    model = torch.compile(model)

# Prepare input prompt
if args.start.startswith('FILE:'):
    with open(args.start[5:], 'r', encoding='utf-8') as f:
        args.start = f.read()
start_ids = encode(args.start)
x = (torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...])

# Generate text
with torch.no_grad():
    with ctx:
        for k in range(args.num_samples):
            y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
            print(decode(y[0].tolist()))
            print('---------------')
