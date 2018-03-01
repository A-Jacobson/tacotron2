import torch
from torch.autograd import Variable

from attention import LocationAttention


def test_attention():
    """
    Attention should output a fixed length context vector (seq len = 1)
    and and a weight for each item in the input sequence
    """
    encoder_out = Variable(torch.zeros(152, 2, 256))  # seq, batch, dim
    query_vector = Variable(torch.zeros(1, 2, 256))  # seq, batch, dim

    attention = LocationAttention(dim=256)
    context, mask = attention(query_vector, encoder_out)
    assert context.size() == (1, 2, 256)  # seq, batch, dim
    assert mask.size() == (2, 152, 1)  # batch, input_seq_len


def test_attention_location():
    encoder_out = Variable(torch.zeros(152, 2, 256))  # seq, batch, dim
    query_vector = Variable(torch.zeros(1, 2, 256))  # seq, batch, dim
    mask = Variable(torch.zeros(2, 152, 1))  # seq, batch, dim
    attention = LocationAttention(dim=256)
    context, mask = attention(query_vector, encoder_out, mask)
    assert context.size() == (1, 2, 256)
    assert mask.size() == (2, 152, 1)  # batch, input_seq_len

