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
spectrogram_pad = -80.0  # change to -80.0? (-80 is the mel value of a window of zeros in the time dim (wave))

# Encoder
num_chars = len(chars) + 1
padding_idx = 0
embedding_dim = 512

# Decoder
# max_iters = 50

# training
teacher_forcing_ratio = 1.0

# SGDR
cycle_length = 2000
min_lr = 1e-5
max_lr = 1e-3
weight_decay = 1e-6  # l2 reg

# max_grad_norm = 10.0

# WavnetVocoder
