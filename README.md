# Tacotron2
NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM
PREDICTIONS
https://arxiv.org/pdf/1712.05884.pdf

WaveNet: A Generative Model for Raw Audio

https://arxiv.org/abs/1609.03499

## Setup

2. install pytorch and torchvision:
```
conda install pytorch -c pytorch
```

3. install tensorflow and tensorboardX for logging.
```
pip install tensorboard
pip install tensorboardX
```

## Contents
- Simple LJ Speech DataLoader
- Mel Spectrogram Prediction network
- [TODO] WaveNet Vocoder (Spectrogram to raw audio)
