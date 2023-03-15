from zeno import (
    distill,
    metric,
    model,
    ZenoOptions,
    DistillReturn,
    MetricReturn,
    ModelReturn,
)
from pandas import DataFrame


@model
def model_ret(name):
    def mod(df: DataFrame, ops: ZenoOptions):
        return ModelReturn(model_output=[""] * len(df))

    return mod


@distill
def length(df: DataFrame, ops: ZenoOptions):
    return DistillReturn(distill_output=df["prompt"].str.len())


@metric
def avg_image_nswf(df: DataFrame, ops: ZenoOptions):
    return MetricReturn(metric=float(df["image_nsfw"].dropna().mean()))


@metric
def avg_prompt_nsfw(df: DataFrame, ops: ZenoOptions):
    return MetricReturn(metric=float(df["prompt_nsfw"].dropna().mean()))
