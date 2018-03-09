import torch
from torch.autograd import Variable

from attention import LocationAttention


def test_attention_sizes():
    """
    Attention should output a fixed length context vector (seq len = 1)
    and and a weight for each item in the input sequence
    """
    encoder_out = Variable(torch.randn(152, 2, 256))  # seq, batch, dim
    query_vector = Variable(torch.randn(1, 2, 1024))  # seq, batch, dim

    attention = LocationAttention(encoded_dim=256, query_dim=1024, attention_dim=128)
    context, mask = attention(query_vector, encoder_out)
    assert context.size() == (1, 2, 128)  # seq, batch, dim
    assert mask.size() == (1, 2, 152)  # seq2, batch, seq1


def test_attention_location_sizes():
    encoder_out = Variable(torch.randn(152, 2, 256))  # seq, batch, dim
    query_vector = Variable(torch.randn(1, 2, 1024))  # seq, batch, dim
    mask = Variable(torch.randn(2, 152, 1))  # seq, batch, dim
    attention = LocationAttention(dim=256)
    context, mask = attention(query_vector, encoder_out, mask)
    assert context.size() == (1, 2, 256)
    assert mask.size() == (1, 2, 152)  # seq2, batch, seq1


def test_attention_location_softmax():
    encoder_out = Variable(torch.randn(152, 2, 256))  # seq, batch, dim
    query_vector = Variable(torch.randn(1, 2, 256))  # seq, batch, dim
    mask = Variable(torch.randn(2, 152, 1))  # seq, batch, dim
    attention = LocationAttention(dim=256)
    context, mask = attention(query_vector, encoder_out, mask)
    assert float(mask[:, 0, :].sum().data) == 1.0  # batch, input_seq_len
