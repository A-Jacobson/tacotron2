from torch.autograd import Variable


def make_stop_targets(targets, audio_lengths):
    #  create stop targets
    # TODO this should be moved to the dataloader.. maybe?
    seq2_len, batch_size, _ = targets.size()
    stop_targets = targets.data.new(batch_size, seq2_len).fill_(0)
    for i in range(len(stop_targets)):
        stop_targets[i, audio_lengths[i] - 1] = 1
    return Variable(stop_targets, requires_grad=False)
