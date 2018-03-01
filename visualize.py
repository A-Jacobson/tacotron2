import matplotlib.pyplot as plt


def show_spectrogram(spec, text=None):
    plt.figure(figsize=(14, 6))
    plt.imshow(spec)
    plt.title(text, fontsize='10')
    plt.colorbar(shrink=0.5, orientation='horizontal')
    plt.ylabel('mels')
    plt.xlabel('frames')

