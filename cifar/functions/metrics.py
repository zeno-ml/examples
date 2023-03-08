from pandas import DataFrame
from sklearn.metrics import f1_score, recall_score
from zeno import ZenoOptions, MetricReturn, metric, distill, DistillReturn


@metric
def accuracy(df, ops: ZenoOptions):
    if len(df) == 0:
        return MetricReturn(metric=0)
    return MetricReturn(
        metric=100 * (df[ops.label_column] == df[ops.output_column]).sum() / len(df)
    )


@metric
def recall(df, ops: ZenoOptions):
    rec = recall_score(df[ops.label_column], df[ops.output_column], average="macro")
    if type(rec) == float:
        return MetricReturn(metric=100 * float(rec))
    else:
        return MetricReturn(metric=0)


@metric
def f1(df, ops: ZenoOptions):
    f = f1_score(df[ops.label_column], df[ops.output_column], average="macro")
    if type(f) == float:
        return MetricReturn(metric=100 * f)
    else:
        return MetricReturn(metric=0)


@distill
def incorrect(df: DataFrame, ops: ZenoOptions):
    return DistillReturn(distill_output=df[ops.label_column] != df[ops.output_column])
