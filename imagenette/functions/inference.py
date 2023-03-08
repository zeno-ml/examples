from zeno import inference, ZenoOptions, InferenceReturn
import gradio as gr


@inference
def gradio_inference(ops: ZenoOptions):
    return InferenceReturn(
        input_components=[gr.Image(type="filepath")],
        output_component=gr.Text(label="Output"),
        input_columns=[ops.data_column],
    )
