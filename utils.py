import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import collate_fn


def train(model, optimizer, dataset, num_epochs, batch_size=1):
    model.train()
    writer = SummaryWriter()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    step = 0
    for epoch in tqdm(range(num_epochs), total=num_epochs, unit=' epochs'):
        total_loss = 0
        pbar = tqdm(loader, total=len(loader), unit=' batches')
        for b, (text_batch, audio_batch, text_lengths, audio_lenghts) in enumerate(pbar):
            text = Variable(text_batch).cuda()
            targets = Variable(audio_batch, requires_grad=False).cuda()

            #  create stop targets
            stop_targets = torch.zeros(targets.size(1), targets.size(0))
            for i in range(len(stop_targets)):
                stop_targets[i, audio_lenghts[i] - 1] = 1
            stop_targets = Variable(stop_targets, requires_grad=False).cuda()

            optimizer.zero_grad()
            outputs, stop_tokens, attention = model(text, targets)
            spec_loss = F.mse_loss(outputs, targets)
            stop_loss = F.binary_cross_entropy_with_logits(stop_tokens, stop_targets)
            loss = spec_loss + stop_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
            pbar.set_description(f'loss: {loss.data[0]}')
            writer.add_scalar('loss', loss.data[0], step)
            if step % 100 == 0:
                writer.add_image('attention', attention[0], step)
                writer.add_image('target', targets.data[0].transpose(0, 1))
                writer.add_image('output', outputs.data[0].transpose(0, 1))
            step += 1
