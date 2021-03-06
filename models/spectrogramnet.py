import torch
from torch import nn
from torch.autograd import Variable

import hyperparams as hp
from attention import LocationAttention


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        return self.dropout(x)


class ConvTanhBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvTanhBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch = nn.BatchNorm1d(out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        return self.tanh(x)


class PreNet(nn.Module):
    """
    Extracts 256d features from 80d input spectrogram frame
    """

    def __init__(self, in_features=80, out_features=256, dropout=0.5):
        super(PreNet, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, previous_y):
        x = self.relu(self.fc1(previous_y))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return x


class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()
        self.conv1 = ConvTanhBlock(in_channels=1, out_channels=512, kernel_size=5, padding=2)
        self.conv2 = ConvTanhBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.conv3 = ConvTanhBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.conv4 = ConvTanhBlock(in_channels=512, out_channels=512, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)


class Encoder(nn.Module):

    def __init__(self, num_chars=hp.num_chars, embedding_dim=512, hidden_size=256):
        super(Encoder, self).__init__()
        self.char_embedding = nn.Embedding(num_embeddings=num_chars,
                                           embedding_dim=embedding_dim, padding_idx=0)
        self.conv1 = ConvBlock(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, padding=2)
        self.conv2 = ConvBlock(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, padding=2)
        self.conv3 = ConvBlock(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=5, padding=2)
        self.birnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True, dropout=0.1) # TODO add soneout

    def forward(self, text):
        # input - (batch, maxseqlen) | (4, 156)
        x = self.char_embedding(text)  # (batch, seqlen, embdim) | (4, 156, 512)
        x = x.permute(0, 2, 1)  # swap to batch, channel, seqlen (4, 512, 156)
        x = self.conv1(x)  # (4, 512, 156)
        x = self.conv2(x)  # (4, 512, 156)
        x = self.conv3(x)  # (4, 512, 156)
        x = x.permute(2, 0, 1)  # swap seq, batch, dim for rnn | (156, 4, 512)
        x, _ = self.birnn(x)  # (156, 4, 512) | 256 dims in either direction
        # sum bidirectional outputs
        x = (x[:, :, :256] + x[:, :, 256:])
        return x


class Decoder(nn.Module):
    """
    Decodes encoder output and previous predicted spectrogram frame into next spectrogram frame.
    """

    def __init__(self, hidden_size=1024, num_layers=2,
                 num_mels=80, num_prenet_features=256):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_mels = num_mels
        
        self.prenet = PreNet(in_features=num_mels, out_features=num_prenet_features)
        self.attention = LocationAttention(encoded_dim=256, query_dim=hidden_size, attention_dim=128)
        self.rnn = nn.LSTM(input_size=num_prenet_features + 256, hidden_size=hidden_size, num_layers=num_layers, dropout=0.1)
        self.spec_out = nn.Linear(in_features=hidden_size + 256, out_features=num_mels)
        self.stop_out = nn.Linear(in_features=hidden_size + 256, out_features=1)
        self.postnet = PostNet()

    def init_hidden(self, batch_size):
        return (nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
                nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())

    def init_mask(self, encoder_out):
        seq1_len, batch_size, _ = encoder_out.size()
        return Variable(encoder_out.data.new(1, batch_size, seq1_len).fill_(0))

    def forward(self, previous_out, encoder_out, decoder_hidden=None, mask=None):
        """
        Decodes a single frame
        """
        previous_out = self.prenet(previous_out)  # (4, 1, 256)
        hidden, cell = decoder_hidden
        context, mask = self.attention(hidden[:-1], encoder_out, mask)
        rnn_input = torch.cat([previous_out, context], dim=2)
        rnn_out, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        spec_frame = self.spec_out(torch.cat([rnn_out, context], dim=2))  # predict next audio frame
        stop_token = self.stop_out(torch.cat([rnn_out, context], dim=2))  # predict stop token
        spec_frame = spec_frame.permute(1, 0, 2)
        spec_frame = spec_frame + self.postnet(spec_frame)  # add residual
        return spec_frame.permute(1, 0, 2), stop_token, decoder_hidden, mask


class MelSpectrogramNet(nn.Module):

    def __init__(self):
        super(MelSpectrogramNet, self).__init__()
        self.encoder = Encoder(num_chars=hp.num_chars)
        self.decoder = Decoder()

    def forward(self, text, decoding_helper):
        encoder_output = self.encoder(text)
        frames, stop_tokens, masks = decoding_helper(self.decoder, encoder_output)
        return frames, stop_tokens, masks
