from zeno import (
    ZenoOptions,
    model,
    distill,
    metric,
    inference,
    ModelReturn,
    InferenceReturn,
    MetricReturn,
    DistillReturn,
)
from pandas import DataFrame
from sklearn.metrics import f1_score, recall_score

from transformers import pipeline
import gradio as gr


@model
def load_model(model_name):
    mod = pipeline("sentiment-analysis", model=model_name)

    def pred(df, ops: ZenoOptions):
        out = mod(list(df[ops.data_column]))
        clean_out = list(map(lambda x: 1 if x["label"] == "POSITIVE" else 0, out))
        return ModelReturn(model_output=clean_out)

    return pred


@inference
def gradio_inference(ops: ZenoOptions):
    return InferenceReturn(
        input_components=[gr.Text(label="Input")],
        output_component=gr.Text(label="Output"),
        input_columns=[ops.data_column],
    )


@distill
def length(df, ops: ZenoOptions):
    return DistillReturn(distill_output=[len(i) for i in df[ops.data_column]])


@distill
def unique_words(df, ops: ZenoOptions):
    return DistillReturn(
        distill_output=[len(set(i.split(" "))) for i in df[ops.data_column]]
    )


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
        return MetricReturn(metric=100 * float(f))
    else:
        return MetricReturn(metric=0)


@distill
def incorrect(df: DataFrame, ops: ZenoOptions):
    return DistillReturn(distill_output=df[ops.label_column] != df[ops.output_column])
