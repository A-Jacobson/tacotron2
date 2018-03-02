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

    Note: modified for bilinear dot product attention instead of bahandanau attention
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