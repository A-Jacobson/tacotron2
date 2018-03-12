import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class TacoTeacher:
    def __init__(self, teacher_forcing_ratio=1.0):
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.targets = None
        self.maxlen = 0

    def set_targets(self, targets):
        self.targets = targets
        self.maxlen = len(targets)

    def __call__(self, decoder, encoder_out):
        seq1_len, batch_size, _ = encoder_out.size()
        outputs = Variable(encoder_out.data.new(self.maxlen, batch_size, decoder.num_mels))
        stop_tokens = Variable(outputs.data.new(self.maxlen, batch_size))
        masks = torch.zeros(self.maxlen, batch_size, seq1_len)

        # start token spectrogram frame of zeros, starting mask of zeros
        output = Variable(outputs.data.new(1, batch_size, decoder.num_mels).fill_(0))
        mask = decoder.init_mask(encoder_out)  # get initial mask
        hidden = decoder.init_hidden(batch_size)
        for t in range(self.maxlen):
            output, stop_token, hidden, mask = decoder(output, encoder_out, hidden, mask)
            outputs[t] = output
            stop_tokens[t] = stop_token
            masks[t] = mask.data
            # teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                output = self.targets[t].unsqueeze(0)
        return outputs, stop_tokens.transpose(1, 0), masks.permute(1, 2, 0)  # batch, src, trg


class TacoGenerator:
    def __init__(self, maxlen=1000, use_stop=False):
        self.maxlen = maxlen
        self.use_stop = use_stop

    def set_maxlen(self, maxlen):
        self.maxlen = maxlen

    def __call__(self, decoder, encoder_out):
        seq1_len, batch_size, _ = encoder_out.size()
        if self.use_stop and batch_size > 1:
            raise ValueError('batching not supported for dynamic stopping')
        outputs = Variable(encoder_out.data.new(self.maxlen, batch_size, decoder.num_mels))
        stop_tokens = Variable(outputs.data.new(self.maxlen, batch_size))
        masks = torch.zeros(self.maxlen, batch_size, seq1_len)

        # start token spectrogram frame of zeros, starting mask of zeros
        output = Variable(outputs.data.new(1, batch_size, decoder.num_mels).fill_(0))
        mask = decoder.init_mask(encoder_out)  # get initial mask
        hidden = decoder.init_hidden(batch_size)
        for t in range(self.maxlen):
            output, stop_token, hidden, mask = decoder(output, encoder_out, hidden, mask)
            outputs[t] = output
            stop_tokens[t] = stop_token
            masks[t] = mask.data
            if self.use_stop and F.sigmoid(stop_token) > 0.5:
                break
            # teacher forcing
        return outputs, stop_tokens.transpose(1, 0), masks.permute(1, 2, 0)  # batch, src, trg
