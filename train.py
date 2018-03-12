import argparse

import torch
from torch.optim import Adam

import hyperparams as hp
from audio import wav_to_spectrogram
from datasets import LJSpeechDataset
from models import MelSpectrogramNet
from sgdr import SGDRScheduler, LRFinderScheduler
from text import text_to_sequence
from utils import train

parser = argparse.ArgumentParser(description='PyTorch Tacotron Spectrogram Training')
parser.add_argument('--data',
                    default='/home/austin/data/tacotron/LJSpeech-1.0',
                    help='path to dataset')
parser.add_argument('--epochs', '-e', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', '-bs', default=12, type=int, help='mini-batch size (default: 12)')
parser.add_argument('--name', '-n', default='melnet', help='experiment name', type=str)
parser.add_argument('--find_lr', default=False, type=bool,
                    help='runs training with LR finding scheduler,'
                         ' check tensorboard plots to choose max_lr')
parser.add_argument('--checkpoint', '-cp', default=None, type=str, help='path to checkpoint')


def main():
    args = parser.parse_args()

    step = 0
    exp_name = f'{args.name}_{hp.max_lr}_{hp.cycle_length}'

    dataset = LJSpeechDataset(path=args.data, text_transforms=text_to_sequence,
                              audio_transforms=wav_to_spectrogram, cache=False)

    model = MelSpectrogramNet()
    if args.checkpoint:
        weights = torch.load(args.checkpoint)
        model.load_state_dict(weights)
        step = int(args.checkpoint.split('/')[-1].split('_')[-1].split('.')[0])
        exp_name = "_".join(args.checkpoint.split('/')[-1].split('_')[:-1])
    model.cuda(device=0)
    optimizer = Adam(model.parameters(), lr=hp.max_lr, weight_decay=hp.weight_decay,
                     betas=(0.9, 0.999), eps=1e-6)
    if args.find_lr:
        scheduler = LRFinderScheduler(optimizer)
    else:
        scheduler = SGDRScheduler(optimizer, min_lr=hp.min_lr,
                                  max_lr=hp.max_lr, cycle_length=hp.cycle_length, current_step=step)

    train(model, optimizer, scheduler,
          dataset, args.epochs, args.batch_size, save_interval=50, exp_name=exp_name, device=0, step=step)


if __name__ == '__main__':
    main()
