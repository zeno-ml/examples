from zeno import model, ModelReturn


@model
def model_wrap(model_name):
    def model_fn(df, ops):
        return ModelReturn(model_output=df["ideal"])

    return model_fn
