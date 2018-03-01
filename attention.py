from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class LocationAttention(nn.Module):
    """
    Calculates context vector based on previous decoder hidden state (query vector),
    encoder output features, and convolutional features extracted from previous attention weights.
    Attention-Based Models for Speech Recognition
    https://arxiv.org/pdf/1506.07503.pdf

    Query vector is either the previous output or the last decoder hidden state.
    """

    def __init__(self, dim, num_location_features=32):
        super(LocationAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=num_location_features,
                              kernel_size=31, padding=15)
        self.W = nn.Linear(dim, dim, bias=False)
        self.L = nn.Linear(num_location_features, dim, bias=False)

    def score(self, query_vector, encoder_out, mask=None):
        # linear transform encoder out (seq, batch, dim)
        encoder_out = self.W(encoder_out)
        # (batch, seq, dim) | (2, 15, 50)
        encoder_out = encoder_out.permute(1, 0, 2)
        if isinstance(mask, Variable):
            conv_features = self.conv(mask.permute(0, 2, 1))  # (batch, dim , seq)
            encoder_out = encoder_out + self.L(conv_features.permute(0, 2, 1))  # (batch, seq , dim)
        # (2, 15, 50) @ (2, 50, 1)
        return encoder_out @ query_vector.permute(1, 2, 0)

    def forward(self, query_vector, encoder_out, mask=None):
        energies = self.score(query_vector, encoder_out)
        mask = F.softmax(energies, dim=1)
        context = encoder_out.permute(
            1, 2, 0) @ mask  # (batch, dim, seq) @ (batch, seq, dim)
        context = context.permute(2, 0, 1)  # (seq, batch, dim)
        return context, mask


# class LocationAttention(nn.Module):
#     """
#     Calculates context vector based on previous decoder hidden state (query vector),
#     encoder output features, and convolutional features extracted from previous attention weights.
#     Attention-Based Models for Speech Recognition
#     https://arxiv.org/pdf/1506.07503.pdf
#     """
#
#     def __init__(self, input_dim, attention_dim, num_location_features=32):
#         super(LocationAttention, self).__init__()
#         self.input_dim = input_dim
#         self.conv = nn.Conv1d(in_channels=1, out_channels=num_location_features,
#                               kernel_size=31, padding=15)
#         self.W = nn.Parameter(torch.Tensor(input_dim * 2, attention_dim))  # (1024, 512)
#         self.U = nn.Parameter(torch.Tensor(input_dim, attention_dim))  # (512, 512)
#         self.L = nn.Parameter(torch.Tensor(num_location_features, attention_dim))  # (32, 512)
#         self.v = nn.Parameter(torch.Tensor(attention_dim, 1))  # (512, 1)
#         self.tanh = nn.Tanh()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.W.size(1))
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, decoder_hidden, encoder_output, attention_weights=None):
#         seq_length, batch_size, _ = encoder_output.size()
#         # make hidden state and encoder output the same length
#         decoder_hidden = decoder_hidden.repeat(seq_length, 1, 1)
#         decoder_hidden = decoder_hidden.contiguous().view(-1, self.input_dim * 2)  # (batch*seq, input_dim)
#         encoder_output = encoder_output.contiguous().view(-1, self.input_dim) # (batch*seq, input_dim)
#         energy_input = decoder_hidden @ self.W + encoder_output @ self.U
#         if isinstance(attention_weights, Variable):
#             conv_features = self.conv(attention_weights.unsqueeze(1))
#             conv_features = conv_features.permute(0, 2, 1).contiguous().view(-1, 32)
#             energy_input += conv_features @ self.L
#         energies = self.tanh(energy_input) @ self.v
#         energies = energies.view(batch_size, -1)  # reshape to (Batch, energies)
#         attention_weights = F.softmax(energies, dim=1)
#         encoder_output = encoder_output.view(batch_size, seq_length, -1)
#         context = attention_weights.unsqueeze(1) @ encoder_output
#         return context.permute(1, 0, 2), attention_weights # seq, batch, dim context
