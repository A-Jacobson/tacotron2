# Text
eos = '~'
pad = '_'
chars = pad + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ' + eos
unk_idx = len(chars)

# Audio
sample_rate = 22050  # hz
fft_frame_size = 50.0  # ms
fft_hop_size = 12.5  # ms
num_mels = 80  # filters
min_freq = 125  # hz
max_freq = 7600  # hz
floor_freq = 0.01  # reference freq for power to db conversion

# Encoder
num_chars = len(chars) + 1
padding_idx = 0
embedding_dim = 512

# Attention

# Decoder
max_iters = 50

# train
teacher_forcing_ratio = 1.0
# WavnetVocoder
