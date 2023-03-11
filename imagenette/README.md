# [Imagenette](https://github.com/fastai/imagenette) Data with [ResNet-50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) Model

This example demonstrates using Zeno with image classification.
Look to the [`config.toml`](config.toml) and `functions/` directory for the details.

Torchhub weights: IMAGENET1K_V1, IMAGENET1K_V2 are loaded and used to predict the classes of the imagenette dataset.

## Run this example in 3 easy steps

### 1. Download python dependencies

```bash
pip3 install zenoml # only if you don't have zeno installed
pip3 install -r requirements.txt
```

### 2. Download the data

```bash
python3 download.py
```

### 3. Run zeno

```bash
zeno config.toml
```

then navigate to `http://localhost:8009` to see the results (you can change this port in the [`config.toml`](config.toml))
