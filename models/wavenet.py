import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    """
    pads the left side of input sequence so the
    kernel does not look into the future.

    Input and output sizes will be the same.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]  # remove trailing padding
        return x


class ResidualLayer(nn.Module):
    """
    A wavenet causal gated residual layer
    """

    def __init__(self, residual_channels, skip_channels, dilation):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_channels, residual_channels,
                                        kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_channels, residual_channels,
                                      kernel_size=2, dilation=dilation)

        self.resconv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skipconv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)
        fx = F.tanh(conv_filter) * F.sigmoid(conv_gate)
        fx = self.resconv(fx)  # restore feature dims with
        skip = self.skipconv(fx)  # conv1x1 goes to skip connections
        residual = fx + x  # residual output goes to next layer
        return skip, residual


class DilatedStack(nn.Module):
    def __init__(self, residual_channels, skip_channels, num_dilations=10):
        """
        Block of dilated residual layers
        Dilation increases exponentially, final dilation will be 2**num_layers.
        a 10 layer block has a receptive field of 1024 (kernel_size * dilation)
        """
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_channels, skip_channels, dilation=2 ** layer)
                          for layer in range(num_dilations)]
        self.residual_stack = nn.ModuleList(residual_stack)

    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
        return torch.cat(skips, dim=0), x  # layers, batch, features, seq


class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet, self).__init__()
        self.embed = nn.Embedding(num_embeddings=256, embedding_dim=256)  # 1, 100, 32 #TODO change to 1hot or MoL
        self.input_conv = CausalConv1d(in_channels=256, out_channels=512, kernel_size=2)
        self.block1 = DilatedStack(residual_channels=512, skip_channels=256, num_dilations=10)
        self.block2 = DilatedStack(residual_channels=512, skip_channels=256, num_dilations=10)
        self.convout = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)

    def forward(self, x):
        x = self.embed(x)  # [1, 100, 32]
        x = x.transpose(1, 2)  # [1, 32, 100]
        x = self.input_conv(x)  # [1, 32, 100]
        skips1, residual = self.block1(x)  # [5, 1, 16, 100] and [1, 32, 100]
        skips2, _ = self.block2(residual)  # [5, 1, 16, 100] and [1, 32, 100]
        skips = torch.cat([skips1, skips2], dim=0)  # [10, 1, 16, 100]
        out = skips.sum(dim=0)  # [1, 16, 100]
        out = F.relu(out)
        out = self.convout(out)  # [1, 256, 100]
        return out

    def generate(self, start, maxlen=100):
        # Naive, original generation #TODO fast wavenet
        outputs = [start]
        for i in range(maxlen):
            probs = self.forward(torch.cat(outputs, dim=1))  # P(all_next|all previous)
            _, output = probs[..., -1:].max(dim=1)  # get prob for the last word and take max idx
            outputs.append(output)
        return torch.cat(outputs[1:])
