from typing import Any, Dict

from pandas_profiling.config import Settings
from pandas_profiling.model.description_variable import CatDescriptionSupervised
from pandas_profiling.report.formatters import fmt, fmt_bytesize, fmt_percent
from pandas_profiling.report.presentation.core import (
    Container,
    Image,
    Table,
    VariableInfo,
)
from pandas_profiling.visualisation.plot import plot_hist_dist, plot_hist_log_odds


def render_date(config: Settings, summary: Dict[str, Any]) -> Dict[str, Any]:
    varid = summary["varid"]
    template_variables = {}

    image_format = config.plot.image_format

    top_items = []
    # Top
    info = VariableInfo(
        summary["varid"],
        summary["varname"],
        "Date",
        summary["alerts"],
        summary["description"],
        style=config.html.style,
    )
    top_items.append(info)

    table1 = Table(
        [
            {
                "name": "Distinct",
                "value": fmt(summary["n_distinct"]),
                "alert": False,
            },
            {
                "name": "Distinct (%)",
                "value": fmt_percent(summary["p_distinct"]),
                "alert": False,
            },
            {
                "name": "Missing",
                "value": fmt(summary["n_missing"]),
                "alert": False,
            },
            {
                "name": "Missing (%)",
                "value": fmt_percent(summary["p_missing"]),
                "alert": False,
            },
            {
                "name": "Memory size",
                "value": fmt_bytesize(summary["memory_size"]),
                "alert": False,
            },
        ],
        style=config.html.style,
    )
    top_items.append(table1)

    table2 = Table(
        [
            {"name": "Minimum", "value": fmt(summary["min"]), "alert": False},
            {"name": "Maximum", "value": fmt(summary["max"]), "alert": False},
        ],
        style=config.html.style,
    )
    top_items.append(table2)

    if config.report.vars.distribution_on_top:
        mini_real_dist = Image(
            plot_hist_dist(config, summary["plot_description"], mini=True, date=True),
            image_format=image_format,
            alt="Mini histogram",
        )
        top_items.append(mini_real_dist)

    if config.report.vars.log_odds_on_top and isinstance(
        summary["plot_description"], CatDescriptionSupervised
    ):
        mini_real_log_odds = Image(
            plot_hist_log_odds(
                config, summary["plot_description"], mini=True, date=True
            ),
            image_format=image_format,
            alt="Mini logo2dds",
        )
        top_items.append(mini_real_log_odds)

    template_variables["top"] = Container(top_items, sequence_type="grid")

    # ==================================================================================

    # distribution
    distribution = Image(
        plot_hist_dist(config, summary["plot_description"], date=True),
        image_format=image_format,
        alt="Distribution histogram",
        name="Distribution",
    )

    # log odds
    if isinstance(summary["plot_description"], CatDescriptionSupervised):
        log_odds = Image(
            plot_hist_log_odds(config, summary["plot_description"], date=True),
            image_format=image_format,
            alt="Mini histogram",
            name="Log Odds",
            caption="Log2 odds with Beta smoothing. alpha + beta = {}".format(
                config.vars.base.smoothing_parameter
            ),
        )
        plots = [distribution, log_odds]
    else:
        plots = [distribution]

    hist = Container(
        plots,
        sequence_type="grid",
        name="Histogram",
        anchor_id=f"{varid}histogram",
    )
    # Bottom
    bottom = Container(
        [hist],
        sequence_type="tabs",
        anchor_id=summary["varid"],
    )

    template_variables["bottom"] = bottom

    return template_variables
