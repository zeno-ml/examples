import os

import torch
import torchvision.transforms as transforms
import time
from PIL import Image
from zeno import model, ModelReturn

from cifar_model import Net

transform_image = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@model
def load_model(model_path):
    net = Net()
    net.load_state_dict(torch.load(model_path))

    def pred(df, ops):
        imgs = [
            Image.open(os.path.join(ops.data_path, img)) for img in df[ops.data_column]
        ]
        imgs = torch.stack([transform_image(img) for img in imgs])  # type: ignore
        with torch.no_grad():
            start = time.time()
            out, emb = net(imgs)
            end = time.time() - start
        return ModelReturn(
            model_output=[
                classes[i] for i in torch.argmax(out, dim=1).detach().numpy()
            ],
            embedding=emb.detach().numpy(),
            other_returns={"time": [end] * len(df)},
        )

    return pred
