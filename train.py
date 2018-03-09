import argparse

from torch.optim import Adam

from audio import wav_to_spectrogram
from datasets import LJSpeechDataset
from models import MelSpectrogramNet
from text import text_to_sequence
from utils import train

parser = argparse.ArgumentParser(description='PyTorch Tacotron Spectrogram Training')
parser.add_argument('-data',
                    default='/home/austin/data/tacotron/LJSpeech-1.0',
                    help='path to dataset')
parser.add_argument('-epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=12, type=int, help='mini-batch size (default: 12)')


def main():
    args = parser.parse_args()

    model = MelSpectrogramNet()
    model.cuda(device=0)
    optimizer = Adam(model.parameters(), lr=1e-3,
                     betas=(0.9, 0.999),
                     eps=1e-6,
                     weight_decay=1e-6)
    dataset = LJSpeechDataset(path=args.data, text_transforms=text_to_sequence,
                              audio_transforms=wav_to_spectrogram, cache=False)
    train(model, optimizer, dataset, args.epochs, args.batch_size, device=0, log_interval=50)


if __name__ == '__main__':
    main()
