from zeno import distill, metric, ZenoOptions, DistillReturn, MetricReturn
from pandas import DataFrame


@distill
def length(df: DataFrame, ops: ZenoOptions):
    return DistillReturn(distill_output=df["prompt"].str.len())


@metric
def avg_image_nswf(df: DataFrame, ops: ZenoOptions):
    return MetricReturn(metric=float(df["image_nsfw"].dropna().mean()))


@metric
def avg_prompt_nsfw(df: DataFrame, ops: ZenoOptions):
    return MetricReturn(metric=float(df["prompt_nsfw"].dropna().mean()))
