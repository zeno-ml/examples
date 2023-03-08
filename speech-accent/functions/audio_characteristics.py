import os

import librosa
import numpy as np
from zeno import ZenoOptions, distill, DistillReturn


@distill
def amplitude(df, ops: ZenoOptions):
    files = [os.path.join(ops.data_path, f) for f in df[ops.data_column]]
    amps = []
    for audio in files:
        y, _ = librosa.load(audio)
        amps.append(float(np.abs(y).mean()))
    return DistillReturn(distill_output=amps)


@distill
def length(df, ops: ZenoOptions):
    files = [os.path.join(ops.data_path, f) for f in df[ops.data_column]]
    amps = []
    for audio in files:
        y, _ = librosa.load(audio)
        amps.append(len(y))
    return DistillReturn(distill_output=amps)
