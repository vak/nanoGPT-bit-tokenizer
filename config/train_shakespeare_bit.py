# train a miniature bit-level shakespeare model
# since we're working with bits, we can use a larger context size
# as each character is represented by 8 bits

out_dir = 'out-shakespeare-bit'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-bit'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_bit'
gradient_accumulation_steps = 1
batch_size = 32  # reduced batch size since sequence length is 8x longer
block_size = 2048  # increased context size since we're working with bits (8x longer sequences)

# larger model since we need to learn bit patterns
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2

# reduced learning rate for continued training
learning_rate = 5e-4  # reduced from 1e-3
max_iters = 5000  # increased from 7000 to 15000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 5e-5  # reduced from 1e-4
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# increased warmup for stability
warmup_iters = 200  # increased from 100

# init_from can be specified via command line:
# --init_from=scratch  - start from scratch
# --init_from=resume   - continue training from checkpoint