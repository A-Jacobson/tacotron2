# Tacotron2
![im](assets/tacotron2.png)

NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM
PREDICTIONS
https://arxiv.org/pdf/1712.05884.pdf

WaveNet: A Generative Model for Raw Audio
https://arxiv.org/abs/1609.03499

## Contents
- Simple LJ Speech DataLoader
- Mel Spectrogram Prediction network (text to Spectrogram)
- [TODO] WaveNet Vocoder (Spectrogram to raw audio)

# Status
- Spectrogram network is functional but not fully trained.
The model takes ~3 hours per epoch on an M6000 gpu.


## Setup

1. install pytorch and torchvision:
```
conda install pytorch -c pytorch
```

2. install other requirements:
```
pip install -r requirements.txt
```
## Usage
train Spectrogram Prediction Network
```
python train.py
```

view logs in Tensorboard
```
tensorboard --logdir runs
```
![im](assets/tensorboard_images.png)

![im](assets/tensorboard_loss.png)




