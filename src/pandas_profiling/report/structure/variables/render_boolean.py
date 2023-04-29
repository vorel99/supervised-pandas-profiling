from typing import List, Union

from pandas_profiling.config import Settings
from pandas_profiling.model.description_variable import (
    CatDescription,
    CatDescriptionSupervised,
)
from pandas_profiling.report.formatters import fmt, fmt_bytesize, fmt_percent
from pandas_profiling.report.presentation.core import (
    Container,
    FrequencyTable,
    FrequencyTableSmall,
    Image,
    Table,
    VariableInfo,
)
from pandas_profiling.report.presentation.core.renderable import Renderable
from pandas_profiling.report.presentation.frequency_table_utils import freq_table
from pandas_profiling.report.structure.variables.render_common import render_common
from pandas_profiling.visualisation.plot import (
    cat_frequency_plot,
    plot_cat_dist,
    plot_cat_log_odds,
)


def render_boolean(config: Settings, summary: dict) -> dict:
    varid = summary["varid"]
    n_obs_bool = config.vars.bool.n_obs
    image_format = config.plot.image_format

    top_items = []

    # Prepare variables
    template_variables = render_common(config, summary)

    # Element composition
    info = VariableInfo(
        anchor_id=summary["varid"],
        alerts=summary["alerts"],
        var_type="Boolean",
        var_name=summary["varname"],
        description=summary["description"],
        style=config.html.style,
    )
    top_items.append(info)

    table = Table(
        [
            {
                "name": "Distinct",
                "value": fmt(summary["n_distinct"]),
                "alert": "n_distinct" in summary["alert_fields"],
            },
            {
                "name": "Distinct (%)",
                "value": fmt_percent(summary["p_distinct"]),
                "alert": "p_distinct" in summary["alert_fields"],
            },
            {
                "name": "Missing",
                "value": fmt(summary["n_missing"]),
                "alert": "n_missing" in summary["alert_fields"],
            },
            {
                "name": "Missing (%)",
                "value": fmt_percent(summary["p_missing"]),
                "alert": "p_missing" in summary["alert_fields"],
            },
            {
                "name": "Memory size",
                "value": fmt_bytesize(summary["memory_size"]),
                "alert": False,
            },
        ],
        style=config.html.style,
    )
    top_items.append(table)

    if config.report.vars.distribution_on_top:
        mini_cat_dist = Image(
            plot_cat_dist(config, summary["plot_description"], mini=True),
            image_format=image_format,
            alt="Mini histogram",
        )
        top_items.append(mini_cat_dist)

    if config.report.vars.log_odds_on_top and isinstance(
        summary["plot_description"], CatDescriptionSupervised
    ):
        mini_cat_log_odds = Image(
            plot_cat_log_odds(config, summary["plot_description"], mini=True),
            image_format=image_format,
            alt="Mini histogram",
        )
        top_items.append(mini_cat_log_odds)

    fqm = FrequencyTableSmall(
        freq_table(
            freqtable=summary["value_counts_without_nan"],
            n=summary["n"],
            max_number_to_print=n_obs_bool,
        ),
        redact=False,
    )

    template_variables["top"] = Container(top_items, sequence_type="grid")

    bottom_items: List[Renderable] = []

    freq_table_bottom = FrequencyTable(
        template_variables["freq_table_rows"],
        name="Common Values (Table)",
        anchor_id=f"{varid}frequency_table",
        redact=False,
    )
    bottom_items.append(freq_table_bottom)

    description: Union[CatDescription, CatDescriptionSupervised] = summary[
        "plot_description"
    ]
    # distribution
    distribution = Image(
        plot_cat_dist(config, description),
        image_format=image_format,
        alt="Histogram",
        name="Distribution",
    )
    # log odds
    if (
        isinstance(description, CatDescriptionSupervised)
        and description.data_col_name != description.target_col_name
    ):
        log_odds = Image(
            plot_cat_log_odds(config, summary["plot_description"]),
            image_format=image_format,
            alt="Log odds",
            name="Log Odds",
            caption="Log2 odds with Beta smoothing. alpha + beta = {}".format(
                config.vars.base.smoothing_parameter
            ),
        )
        plots = [distribution, log_odds]
    else:
        plots = [distribution]

    bottom_items.append(
        Container(
            plots,
            sequence_type="grid",
            name="Visualization",
            anchor_id=f"{varid}histogram2",
        )
    )

    show = config.plot.cat_freq.show
    max_unique = config.plot.cat_freq.max_unique

    if show and (max_unique > 0):
        cat_frequency_plot_container = None
        if isinstance(summary["value_counts_without_nan"], list):
            cat_frequency_plot_container = Container(
                [
                    Image(
                        cat_frequency_plot(
                            config,
                            s,
                        ),
                        image_format=image_format,
                        alt=config.html.style._labels[idx],
                        name=config.html.style._labels[idx],
                        anchor_id=f"{varid}cat_frequency_plot_{idx}",
                    )
                    for idx, s in enumerate(summary["value_counts_without_nan"])
                ],
                anchor_id=f"{varid}cat_frequency_plot",
                name="Common Values (Plot)",
                sequence_type="batch_grid",
                batch_size=len(config.html.style._labels),
            )

        else:
            cat_frequency_plot_container = Image(
                cat_frequency_plot(
                    config,
                    summary["value_counts_without_nan"],
                ),
                image_format=image_format,
                alt="Common Values (Plot)",
                name="Common Values (Plot)",
                anchor_id=f"{varid}cat_frequency_plot",
            )

        bottom_items.append(cat_frequency_plot_container)

    template_variables["bottom"] = Container(
        bottom_items, sequence_type="tabs", anchor_id=f"{varid}bottom"
    )

    return template_variables
