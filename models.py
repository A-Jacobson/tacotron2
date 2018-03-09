import random

import torch
from torch import nn

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

    def __init__(self, num_chars=hp.num_chars):
        super(Encoder, self).__init__()
        self.char_embedding = nn.Embedding(num_embeddings=num_chars,
                                           embedding_dim=512, padding_idx=0)
        self.conv1 = ConvBlock(512, 512, 5, 2)
        self.conv2 = ConvBlock(512, 512, 5, 2)
        self.conv3 = ConvBlock(512, 512, 5, 2)
        self.birnn = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, dropout=0.1)

    def forward(self, text):
        # input - (batch, maxseqlen) | (4, 156)
        x = self.char_embedding(text)  # (batch, seqlen, embdim) | (4, 156, 512)
        x = x.permute(0, 2, 1)  # swap to batch, channel, seqlen (4, 512, 156)
        x = self.conv1(x)  # (4, 512, 156)
        x = self.conv2(x)  # (4, 512, 156)
        x = self.conv3(x)  # (4, 512, 156)
        x = x.permute(2, 0, 1)  # swap seq, batch, dim for rnn | (156, 4, 512)
        x, hidden = self.birnn(x)  # (156, 4, 512) | 256 dims in either direction
        # sum bidirectional outputs
        x = (x[:, :, :256] + x[:, :, 256:])
        return x, hidden


class Decoder(nn.Module):
    """
    Decodes encoder output and previous predicted spectrogram frame into next spectrogram frame.
    """

    def __init__(self, hidden_size=1024, num_layers=2):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.prenet = PreNet(in_features=80, out_features=256)
        # self.project_encoded = nn.Linear(256, 128, bias=False)
        # self.project_query = nn.Linear(hidden_size, 128, bias=False)
        self.attention = LocationAttention(encoded_dim=256, query_dim=1024, attention_dim=128)
        self.rnn = nn.LSTM(input_size=256 + 256, hidden_size=hidden_size, num_layers=num_layers, dropout=0.1)
        self.spec_out = nn.Linear(in_features=1024 + 256, out_features=80)
        self.stop_out = nn.Linear(in_features=1024 + 256, out_features=1)
        self.postnet = PostNet()

    def init_hidden(self, batch_size):
        return (nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
                nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())

    def _forward(self, previous_out, encoder_out, decoder_hidden=None, mask=None):
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

    def forward(self, encoder_out, targets, teacher_forcing_ratio=0.5):
        outputs = []
        stop_tokens = []
        masks = []

        start_token = torch.zeros_like(targets[:1])
        hidden = self.init_hidden(encoder_out.size(1))
        output, stop_token, hidden, mask = self._forward(start_token,
                                                         encoder_out, hidden)
        for t in range(len(targets)):
            output, stop, hidden, mask = self._forward(output.detach(),
                                                       encoder_out,
                                                       hidden,
                                                       mask)
            outputs.append(output)
            stop_tokens.append(stop)
            masks.append(mask.data)
            teacher = random.random() < teacher_forcing_ratio
            if teacher:
                output = targets[t].unsqueeze(0)

        outputs = torch.cat(outputs)
        stop_tokens = torch.cat(stop_tokens)
        masks = torch.cat(masks)  # seq2, batch, seq1

        stop_tokens = stop_tokens.transpose(1, 0).squeeze()
        if len(stop_tokens.size()) == 1:
            stop_tokens = stop_tokens.unsqueeze(0)
        return outputs, stop_tokens, masks.permute(1, 2, 0)  # batch, src, trg


class MelSpectrogramNet(nn.Module):

    def __init__(self):
        super(MelSpectrogramNet, self).__init__()
        self.encoder = Encoder(num_chars=hp.num_chars)
        self.decoder = Decoder()

    def forward(self, text, targets, teacher_forcing_ratio=hp.teacher_forcing_ratio):
        encoder_output, _ = self.encoder(text)
        outputs, stop_tokens, masks = self.decoder(encoder_output,
                                                   targets,
                                                   teacher_forcing_ratio)
        return outputs, stop_tokens, masks
