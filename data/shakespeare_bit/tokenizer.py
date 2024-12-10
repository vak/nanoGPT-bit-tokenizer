"""
Bit-level tokenizer for Shakespeare dataset.
Provides functions for encoding and decoding text to/from bits.
"""

def char_to_bits(c):
    """Convert a character to its 8-bit binary representation"""
    return [int(b) for b in format(ord(c), '08b')]

def bits_to_char(bits):
    """Convert 8 bits back to a character"""
    assert len(bits) == 8
    return chr(int(''.join(map(str, bits)), 2))

def encode(s):
    """Convert a string to a list of bits"""
    result = []
    for c in s:
        result.extend(char_to_bits(c))
    return result

def decode(l):
    """Convert a list of bits back to a string"""
    assert len(l) % 8 == 0, "Bit sequence length must be a multiple of 8"
    chars = []
    for i in range(0, len(l), 8):
        chars.append(bits_to_char(l[i:i+8]))
    return ''.join(chars) 