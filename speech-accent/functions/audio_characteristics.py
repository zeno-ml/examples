import os

import librosa
import numpy as np

from zeno import DistillReturn, ZenoOptions, distill


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


@distill
def special_chars(df, ops: ZenoOptions):
    chars = df[ops.output_column].str.count(r"[^\w\s]")
    return DistillReturn(distill_output=chars)
