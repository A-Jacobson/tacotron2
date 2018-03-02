from torch.optim import Adam

from audio import wav_to_spectrogram
from datasets import LJSpeechDataset
from models import MelSpectrogramNet
from text import text_to_sequence
from utils import train
from torch.utils.data.dataset import Subset


num_epochs = 10
batch_size = 1

PATH = '/home/austin/data/tacotron/LJSpeech-1.0'
dataset = LJSpeechDataset(path=PATH, text_transforms=text_to_sequence,
                          audio_transforms=wav_to_spectrogram)

dataset = Subset(dataset, range(1))

melnet = MelSpectrogramNet()
melnet.cuda()
optimizer = Adam(melnet.parameters())
train(melnet, optimizer, dataset, num_epochs, batch_size)
