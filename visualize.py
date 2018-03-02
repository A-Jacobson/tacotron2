import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image


def show_spectrogram(spec, text=None, return_array=False):
    plt.figure(figsize=(14, 6))
    plt.imshow(spec)
    plt.title(text, fontsize='10')
    plt.colorbar(shrink=0.5, orientation='horizontal')
    plt.ylabel('mels')
    plt.xlabel('frames')
    if return_array:
        plt.tight_layout()
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        return np.array(Image.open(buff))


def show_attention(attention, return_array=False):
    plt.figure(figsize=(14, 6))
    plt.imshow(attention)
    plt.ylabel('text sequence')
    plt.xlabel('spectrogram frame')
    if return_array:
        plt.tight_layout()
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        return np.array(Image.open(buff))



