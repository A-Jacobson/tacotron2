import torch
from torch.optim import Adam

from audio import wav_to_spectrogram
from datasets import LJSpeechDataset
from models import MelSpectrogramNet
from text import text_to_sequence
from utils import train

num_epochs = 1
batch_size = 1

PATH = '/home/austin/data/tacotron2/LJSpeech-1.0'
dataset = LJSpeechDataset(path=PATH, text_transforms=text_to_sequence,
                          audio_transforms=wav_to_spectrogram)

melnet = MelSpectrogramNet()
melnet.cuda()
optimizer = Adam(melnet.parameters())
train(melnet, optimizer, dataset, num_epochs, batch_size)

torch.save(melnet.state_dict(), 'checkpoints/melnet.pt')
