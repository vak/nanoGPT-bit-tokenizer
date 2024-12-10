# tiny shakespeare, bit-level

Tiny shakespeare dataset processed at the bit level. Each character is converted to its 8-bit ASCII representation.

After running `prepare.py`:

- Each character in the original text is converted to 8 bits
- The vocabulary size is 2 (binary: 0 and 1)
- train.bin will have ~8M bits (8 times the number of characters)
- val.bin will have ~890K bits

This representation allows the model to potentially learn patterns at the bit level, which might be interesting for compression or understanding how the model learns to reconstruct characters from binary patterns. 