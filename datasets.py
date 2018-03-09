import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import hyperparams as hp
from audio import load_wav, wav_to_spectrogram


class LJSpeechDataset(Dataset):

    def __init__(self, path, text_transforms=None, audio_transforms=None, cache=False):
        self.path = path
        self.metadata = pd.read_csv(f'{path}/metadata.csv', sep='|',
                                    names=['wav', 'transcription', 'text'],
                                    usecols=['wav', 'text'])
        self.metadata.dropna(inplace=True)
        self.audio_transforms = audio_transforms
        self.text_transforms = text_transforms
        self.cache = cache
        if cache:
            self.cache_spectrograms()

    def cache_spectrograms(self):
        wav_filenames = self.metadata['wav']
        spectrograms_path = f'{self.path}/spectrograms'
        if not os.path.exists(spectrograms_path):
            os.makedirs(spectrograms_path)
            print('Building Cache..')
            for name in tqdm(wav_filenames, total=len(wav_filenames)):
                audio, _ = load_wav(f'{self.path}/wavs/{name}.wav')
                S = self.audio_transforms(audio)
                np.save(f'{spectrograms_path}/{name}.npy', S)

    def __getitem__(self, index):
        text = self.metadata.iloc[index]['text']
        filename = self.metadata.iloc[index]['wav']
        if self.text_transforms:
            text = self.text_transforms(text)
        if self.cache:
            audio = np.load(f'{self.path}/spectrograms/{filename}.npy')
            return text, audio

        audio, _ = load_wav(f'{self.path}/wavs/{filename}.wav')
        if self.audio_transforms:
            audio = self.audio_transforms(audio)
        return text, audio

    def __len__(self):
        return len(self.metadata)


class WaveNetDataset(Dataset):
    """
    loads spectrogram and raw audio pairs for testing wavnet

    real spectrogramnet outputs much be cached and used in a separate dataset
    """

    def __init__(self, path):
        self.path = path
        self.metadata = pd.read_csv(f'{path}/metadata.csv', sep='|',
                                    names=['wav', 'transcription', 'text'],
                                    usecols=['wav', 'text'])
        self.metadata.dropna(inplace=True)

    def __getitem__(self, index):
        wav_filename = self.metadata.iloc[index]['wav']
        audio, _ = load_wav(f'{self.path}/wavs/{wav_filename}.wav')
        S = wav_to_spectrogram(audio)
        return S, audio

    def __len__(self):
        return len(self.metadata)


def wav_collate(batch):
    spec = [item[0] for item in batch]
    audio = [item[1] for item in batch]

    spec_lengths = [len(x) for x in spec]
    audio_lengths = [len(x) for x in audio]

    max_spec = max(spec_lengths)
    max_audio = max(audio_lengths)

    spec_batch = np.stack(pad2d(x, max_spec) for x in spec)
    audio_batch = np.stack(pad1d(x, max_audio) for x in audio)

    return (torch.FloatTensor(spec_batch).permute(0, 2, 1),  # (batch, channel, time)
            torch.FloatTensor(audio_batch),
            spec_lengths, audio_lengths)


def collate_fn(batch):
    """
    Pads Variable length sequence to size of longest sequence.
    Args:
        batch:

    Returns: Padded sequences and original sizes.

    """
    text = [item[0] for item in batch]
    audio = [item[1] for item in batch]

    text_lengths = [len(x) for x in text]
    audio_lengths = [len(x) for x in audio]

    max_text = max(text_lengths)
    max_audio = max(audio_lengths)

    text_batch = np.stack(pad1d(x, max_text) for x in text)
    audio_batch = np.stack(pad2d(x, max_audio) for x in audio)

    return (torch.LongTensor(text_batch),
            torch.FloatTensor(audio_batch).permute(1, 0, 2),
            text_lengths, audio_lengths)


def pad1d(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=hp.padding_idx)


def pad2d(seq, max_len, dim=80):
    padded = np.zeros((max_len, dim))
    padded[:len(seq), :] = seq
    return padded
