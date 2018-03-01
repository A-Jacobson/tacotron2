import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from audio import load_wav


class LJSpeechDataset(Dataset):

    def __init__(self, path, text_transforms=None, audio_transforms=None):
        self.path = path
        self.metadata = pd.read_csv(f'{path}/metadata.csv', sep='|',
                                    names=['wav', 'transcription', 'text'],
                                    usecols=['wav', 'text'])
        self.metadata.dropna(inplace=True)
        self.audio_transforms = audio_transforms
        self.text_transforms = text_transforms

    def __getitem__(self, index):
        text = self.metadata.iloc[index]['text']
        wav_filename = self.metadata.iloc[index]['wav']
        audio, _ = load_wav(f'{self.path}/wavs/{wav_filename}.wav')
        if self.text_transforms:
            text = self.text_transforms(text)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)
        return text, audio

    def __len__(self):
        return len(self.metadata)


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

    text_batch = np.stack(pad_text(x, max_text) for x in text)
    audio_batch = np.stack(pad_spectrogram(x, max_audio) for x in audio)

    return (torch.LongTensor(text_batch),
            torch.FloatTensor(audio_batch).permute(1, 0, 2),
            text_lengths, audio_lengths)


def pad_text(text, max_len):
    return np.pad(text, (0, max_len - len(text)), mode='constant', constant_values=0)


def pad_spectrogram(S, max_len):
    padded = np.ones((max_len, 80)) * -80
    padded[:len(S), :] = S
    return padded
