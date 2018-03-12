import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from tqdm import tqdm

import hyperparams as hp
from datasets import collate_fn
from decoding_helpers import TacoTeacher
from text import sequence_to_text
from visualize import show_spectrogram, show_attention


def train(model, optimizer, scheduler, dataset, num_epochs, batch_size=1,
          save_interval=50, exp_name='melnet', device=1, step=0):
    model.train()
    writer = SummaryWriter(f'runs/{exp_name}')
    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_fn, pin_memory=True,
                        num_workers=6, shuffle=True)
    tacoteacher = TacoTeacher()
    for _ in tqdm(range(num_epochs), total=num_epochs, unit=' epochs'):
        pbar = tqdm(loader, total=len(loader), unit=' batches')
        for b, (text_batch, audio_batch, text_lengths, audio_lengths) in enumerate(pbar):
            text = Variable(text_batch).cuda(device)
            targets = Variable(audio_batch, requires_grad=False).cuda(device)
            stop_targets = make_stop_targets(targets, audio_lengths)
            tacoteacher.set_targets(targets)
            outputs, stop_tokens, attention = model(text, tacoteacher)
            spec_loss = F.mse_loss(outputs, targets)
            stop_loss = F.binary_cross_entropy_with_logits(stop_tokens, stop_targets)
            loss = spec_loss + stop_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), hp.max_grad_norm, norm_type=2)  # prevent exploding grads
            scheduler.step()
            optimizer.step()
            pbar.set_description(f'loss: {loss.data[0]:.4f}')
            writer.add_scalar('loss', loss.data[0], step)
            writer.add_scalar('lr', scheduler.lr, step)

            if step % save_interval == 0:
                torch.save(model.state_dict(), f'checkpoints/{exp_name}_{str(step)}.pt')

                # plot the first sample in the batch
                attention_plot = show_attention(attention[0], return_array=True)
                output_plot = show_spectrogram(outputs.data.permute(1, 2, 0)[0],
                                               sequence_to_text(text.data[0]),
                                               return_array=True)
                target_plot = show_spectrogram(targets.data.permute(1, 2, 0)[0],
                                               sequence_to_text(text.data[0]),
                                               return_array=True)

                writer.add_image('attention', attention_plot, step)
                writer.add_image('output', output_plot, step)
                writer.add_image('target', target_plot, step)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), step, bins='doane')
            step += 1


def make_stop_targets(targets, audio_lengths):
    #  create stop targets
    # TODO this should be moved to the dataloader.. maybe?
    seq2_len, batch_size, _ = targets.size()
    stop_targets = targets.data.new(batch_size, seq2_len).fill_(0)
    for i in range(len(stop_targets)):
        stop_targets[i, audio_lengths[i] - 1] = 1
    return Variable(stop_targets, requires_grad=False)
