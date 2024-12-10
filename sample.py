"""
Sample from a trained model
"""
import os
import pickle
import torch
from model import GPTConfig, GPT
import random

# -----------------------------------------------------------------------------
# configuration with defaults
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start_text = "KING "  # default value, can be overridden by command line
num_samples = 10 # number of samples to draw
max_new_tokens = 496 # adjusted to be a multiple of 8 (for bit to char conversion)
temperature = 0.8 # 1.0 = no change, < 1.0 = more conservative, > 1.0 = more creative
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# override from command line
exec(open('configurator.py').read())
print(f'Using start text: "{start_text}"')

# Convert text to bits
start = ''.join(format(ord(c), '08b') for c in start_text)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in the dataset directory
load_meta = False
meta_path = None
meta = None
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    raise ValueError("Meta file not found! This script requires bit-level meta information.")

def bits_to_char(bits):
    """Convert 8 bits back to a character"""
    assert len(bits) % 8 == 0, "Bit sequence length must be a multiple of 8"
    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        char_val = int(''.join(str(b) for b in byte_bits), 2)
        chars.append(chr(char_val))
    return ''.join(chars)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# ensure max_new_tokens is a multiple of 8
max_new_tokens = (max_new_tokens // 8) * 8

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            bit_sequence = [int(itos[i]) for i in y[0].tolist()]
            # ensure the sequence length is a multiple of 8 by truncating if necessary
            bit_sequence = bit_sequence[:(len(bit_sequence) // 8) * 8]
            print(bits_to_char(bit_sequence))
            print('---------------')
