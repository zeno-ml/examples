import numpy as np
from pandas import DataFrame
from sklearn.metrics import f1_score, recall_score

from zeno import DistillReturn, MetricReturn, ZenoOptions, distill, metric


@metric
def accuracy(df, ops: ZenoOptions):
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(
        metric=100 * (df[ops.label_column] == df[ops.output_column]).sum() / len(df)
    )


@metric
def recall(df, ops: ZenoOptions):
    if len(df) == 0:
        return MetricReturn(metric=0)

    rec = recall_score(
        df[ops.label_column], df[ops.output_column], average="macro", zero_division=0
    )
    if type(rec) == np.float64:
        return MetricReturn(metric=100 * rec)
    else:
        return MetricReturn(metric=0)


@metric
def f1(df, ops: ZenoOptions):
    if len(df) == 0:
        return MetricReturn(metric=0)
    
    f = f1_score(
        df[ops.label_column], df[ops.output_column], average="macro", zero_division=0
    )
    if type(f) == np.float64:
        return MetricReturn(metric=100 * f)
    else:
        return MetricReturn(metric=0)


@distill
def incorrect(df: DataFrame, ops: ZenoOptions):
    return DistillReturn(distill_output=df[ops.label_column] != df[ops.output_column])


# @distill
# def correct(df: DataFrame, ops: ZenoOptions):
#     return DistillReturn(distill_output=df[ops.label_column] == df[ops.output_column])
