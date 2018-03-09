import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class LocationAttention(nn.Module):
    """
    Calculates context vector based on previous decoder hidden state (query vector),
    encoder output features, and convolutional features extracted from previous attention weights.
    Attention-Based Models for Speech Recognition
    https://arxiv.org/pdf/1506.07503.pdf
    """

    def __init__(self, encoded_dim, query_dim, attention_dim, num_location_features=32):
        super(LocationAttention, self).__init__()
        self.f = nn.Conv1d(in_channels=1, out_channels=num_location_features,
                           kernel_size=31, padding=15)
        self.U = nn.Linear(num_location_features, attention_dim)
        self.W = nn.Linear(query_dim, attention_dim)
        self.V = nn.Linear(encoded_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def score(self, query_vector, encoder_out, mask=None):
        encoder_out = self.V(encoder_out)  # (seq, batch, atten_dim) # project to attn dim
        query_vector = self.W(query_vector)  # (seq, batch, atten_dim)
        attention_input = encoder_out + query_vector
        if isinstance(mask, Variable):
            attention_input += self.f(mask)
        return self.w(self.tanh(attention_input))

    def forward(self, query_vector, encoder_out, mask=None):
        energies = self.score(query_vector, encoder_out)
        mask = F.softmax(energies, dim=0)
        context = encoder_out.permute(1, 2, 0) @ mask.permute(1, 0, 2)  # (batch, seq1, seq2)
        context = context.permute(2, 0, 1)  # (seq2, batch, encoder_dim)
        mask = mask.permute(2, 1, 0)  # (seq2, batch, seq1)
        return context, mask
