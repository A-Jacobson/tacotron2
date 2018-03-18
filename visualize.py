import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


def show_spectrogram(spec, text=None, return_array=False):
    sns.reset_orig()
    plt.figure(figsize=(14, 6))
    plt.imshow(spec)
    if text:
        plt.title(text, fontsize='10')
    plt.colorbar(shrink=0.5, orientation='horizontal')
    plt.ylabel('mels')
    plt.xlabel('frames')
    if return_array:
        plt.tight_layout()
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        plt.close()
        buff.seek(0)
        return np.array(Image.open(buff))
    plt.close()


def show_audio(audio, text=None, return_array=False):
    sns.reset_orig()
    plt.figure(figsize=(14, 3))
    plt.plot(audio, linewidth=0.08, alpha=0.7)
    if text:
        plt.title(text, fontsize='10')
    plt.ylabel('amplitude')
    plt.xlabel('frames')
    if return_array:
        plt.tight_layout()
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        plt.close()
        buff.seek(0)
        return np.array(Image.open(buff))
    plt.close()


def show_attention(attention, return_array=False):
    plt.figure(figsize=(14, 6))
    sns.heatmap(attention,
                xticklabels=20,
                yticklabels=10,
                cmap="Blues")
    plt.ylabel('Source (Characters)')
    plt.xlabel('Prediction (Spectrogram Frames)')
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)
    if return_array:
        plt.tight_layout()
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        plt.close()
        buff.seek(0)
        return np.array(Image.open(buff))
    plt.close()

