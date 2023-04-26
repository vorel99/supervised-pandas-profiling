from pandas_profiling.config import Settings
from pandas_profiling.model.model import ModelData, ModelModule
from pandas_profiling.report.formatters import fmt_percent
from pandas_profiling.report.presentation.core.container import Container
from pandas_profiling.report.presentation.core.table import Table


def render_model_evaluation(config: Settings, model: ModelData, name: str) -> Container:
    model_evaluation = model.evaluate()
    items = []
    table = Table(
        [
            {
                "name": "Accuracy (%)",
                "value": fmt_percent(model_evaluation.accuracy),
            },
            {
                "name": "Precision (%)",
                "value": fmt_percent(model_evaluation.precision),
            },
            {
                "name": "Recall (%)",
                "value": fmt_percent(model_evaluation.recall),
            },
            {
                "name": "F1 score (%)",
                "value": fmt_percent(model_evaluation.f1_score),
            },
        ],
        style=config.html.style,
    )
    items.append(table)

    return Container(
        items, name=name, sequence_type="grid", anchor_id="model_tab_{}".format(name)
    )


def render_model_module(config: Settings, model_module: ModelModule) -> Container:
    items = []

    def_model_tab = render_model_evaluation(
        config, model_module.default_model, "Base model"
    )
    items.append(def_model_tab)

    if model_module.transformed_model:
        trans_model_tab = render_model_evaluation(
            config, model_module.transformed_model, "Model with transformed data"
        )
        items.append(trans_model_tab)

    return Container(
        items,
        name="Model",
        sequence_type="tabs",
        anchor_id="model_module",
    )
