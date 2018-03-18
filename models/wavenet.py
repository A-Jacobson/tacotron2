import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class OneHot(nn.Module):
    """
    One hot encodes a categorical sequence
    functions similar to an embedding layer
    """

    def __init__(self, num_categories=256):
        super(OneHot, self).__init__()
        self.num_categories = num_categories

    def forward(self, indices):
        indices = indices.data
        batch_size, sequence_length = indices.size()
        one_hot = indices.new(batch_size, self.num_categories, sequence_length).zero_()
        one_hot.scatter_(1, indices.unsqueeze(dim=1), 1)
        return Variable(one_hot.float())


class CausalConv1d(nn.Module):
    """
    pads the left side of input sequence just enough so that the convolutional kernel
    does not look into the future.

    Input and output sizes will be the same.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.conv1.padding[0]]  # remove trailing padding
        return x


class ResidualLayer(nn.Module):
    """A wavenet causal gated residual layer that supports local conditioning.
    """

    def __init__(self, residual_channels, skip_channels, dilation, conditioning_channels=None):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_channels, residual_channels,
                                        kernel_size=3, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_channels, residual_channels,
                                      kernel_size=3, dilation=dilation)
        self.resconv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skipconv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

        if conditioning_channels:
            self.conditioning_conv1x1 = nn.Conv1d(conditioning_channels, residual_channels * 2, kernel_size=1)

    def forward(self, x, local_conditioning=None):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)

        if local_conditioning is not None:
            local_conditioning = self.conditioning_conv1x1(local_conditioning)
            local_conditioning_filter, local_conditioning_gate = local_conditioning.chunk(chunks=2, dim=1)
            conv_filter += local_conditioning_filter
            conv_gate += local_conditioning_gate

        fx = F.tanh(conv_filter) * F.sigmoid(conv_gate)
        fx = self.resconv(fx)  # restore feature dims with
        skip = self.skipconv(fx)  # conv1x1 goes to skip connections
        residual = fx + x  # residual output goes to next layer
        return skip, residual


class DilatedStack(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation_depth, conditioning_channels=None):
        """
        Block of dilated residual layers
        Dilation increases exponentially, final dilation will be 2**num_layers.
        a 10 layer block has a receptive field of 1024 (kernel_size * dilation).
        """
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_channels, skip_channels, 2 ** layer, conditioning_channels)
                          for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)

    def forward(self, x, local_conditioning=None):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x, local_conditioning)
            skips.append(skip.unsqueeze(0))
        return torch.cat(skips, dim=0), x  # layers, batch, features, seq


class UpsamplingNetwork(nn.Module):
    """Upsamples local conditioning variables to the same length as input.

    scales = [16, 17] for 275x upsampling?
    """

    def __init__(self, conditioning_channels, scales):
        super(UpsamplingNetwork, self).__init__()
        upsample_layers = [nn.ConvTranspose1d(conditioning_channels, conditioning_channels,
                                              kernel_size=3, stride=scale, padding=1, output_padding=1)
                           for scale in scales]
        self.upsample_layers = nn.ModuleList(upsample_layers)

    def forward(self, x):
        for upsample in self.upsample_layers:
            x = upsample(x)
        return x


class WaveNet(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation_cycles, dilation_depth, conditioning_channels=None,
                 upsampling_scales=None):
        super(WaveNet, self).__init__()
        self.conditioning_channels = conditioning_channels
        self.upsampling_scales = upsampling_scales
        self.one_hot_encode = OneHot(num_categories=256)
        self.input_conv = CausalConv1d(in_channels=256, out_channels=residual_channels, kernel_size=3)
        self.dilated_stacks = nn.ModuleList(
                [DilatedStack(residual_channels, skip_channels, dilation_depth, conditioning_channels)
                 for cycle in range(dilation_cycles)]
                )
        self.convout_1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        self.convout_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)

        if conditioning_channels and upsampling_scales:
            self.local_conditioning_upsample = UpsamplingNetwork(conditioning_channels,
                                                                 upsampling_scales)

    def forward(self, x, local_conditioning=None):
        x = self.one_hot_encode(x)  # [batch, num_mu, seq_len]
        x = self.input_conv(x)  # [1, 256, 100] expects one_hot encoded inputs

        if local_conditioning is not None:
            if not self.conditioning_channels and not self.upsampling_scales:
                raise ValueError('set `conditioning_channels`, and `upsampling_scales` to use local_conditioning.')
            local_conditioning = self.local_conditioning_upsample(local_conditioning)

        skip_connections = []
        for cycle in self.dilated_stacks:
            skips, x = cycle(x, local_conditioning)  # [5, 1, 16, 100] and [1, 32, 100]
            skip_connections.append(skips)
        skip_connections = torch.cat(skip_connections, dim=0)  # [10, 1, 16, 100]

        # gather all output skip connections to generate output, discard last residual output
        out = skip_connections.sum(dim=0)  # [1, 16, 100]
        out = F.relu(out)
        out = self.convout_1(out)  # [1, 256, 100]
        out = F.relu(out)
        return self.convout_2(out)

    def generate(self, start, maxlen=100):
        outputs = [start]
        for i in range(maxlen):
            probs = self.forward(torch.cat(outputs, dim=1))  # P(all_next|all previous)
            _, output = probs[..., -1:].max(dim=1)  # get prob for the last word and take max idx
            outputs.append(output)
        return torch.cat(outputs, dim=1)[..., start.size(dim=1):]
