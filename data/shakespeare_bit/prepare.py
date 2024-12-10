"""
Prepare the Shakespeare dataset for bit-level language modeling.
Each character is converted to its ASCII value and then to its binary representation.
Will save train.bin, val.bin containing the ids (0s and 1s), and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

def char_to_bits(c):
    """Convert a character to its 8-bit binary representation"""
    return [int(b) for b in format(ord(c), '08b')]

def bits_to_char(bits):
    """Convert 8 bits back to a character"""
    assert len(bits) == 8
    return chr(int(''.join(map(str, bits)), 2))

# Convert text to bits
bits = []
for c in data:
    bits.extend(char_to_bits(c))

# create the train and test splits
n = len(bits)
train_data = bits[:int(n*0.9)]
val_data = bits[int(n*0.9):]

# encode both to integers (though they're already integers 0/1)
train_ids = train_data
val_ids = val_data
print(f"train has {len(train_ids):,} bits")
print(f"val has {len(val_ids):,} bits")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint8)
val_ids = np.array(val_ids, dtype=np.uint8)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information with stoi and itos mappings
meta = {
    'vocab_size': 2,  # binary, so only 0 and 1
    'stoi': {'0': 0, '1': 1},  # string to integer mapping
    'itos': {0: '0', 1: '1'},  # integer to string mapping
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f) 