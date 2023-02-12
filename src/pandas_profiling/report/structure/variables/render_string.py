from pandas_profiling.report.formatters import fmt, fmt_bytesize, fmt_percent
from pandas_profiling.report.presentation.core.container import Container
from pandas_profiling.report.presentation.core.image import Image
from pandas_profiling.report.presentation.core.table import Table
from pandas_profiling.report.structure.variables.base_render import \
    BaseRenderVariable
from pandas_profiling.visualisation.plot import plot_word_cloud


class RenderString(BaseRenderVariable):
    def name(self) -> str:
        return "String"

    def _get_top(self):
        """Render top of string variable.

        Contains
        - basic info
        - table with information about missing values
        - word cloud graph"""
        info = self._get_info()
        table = Table(
            [
                {
                    "name": "Distinct",
                    "value": fmt(self.summary["n_distinct"]),
                    "alert": "n_distinct" in self.summary["alert_fields"],
                },
                {
                    "name": "Distinct (%)",
                    "value": fmt_percent(self.summary["p_distinct"]),
                    "alert": "p_distinct" in self.summary["alert_fields"],
                },
                {
                    "name": "Missing",
                    "value": fmt(self.summary["n_missing"]),
                    "alert": "n_missing" in self.summary["alert_fields"],
                },
                {
                    "name": "Missing (%)",
                    "value": fmt_percent(self.summary["p_missing"]),
                    "alert": "p_missing" in self.summary["alert_fields"],
                },
                {
                    "name": "Memory size",
                    "value": fmt_bytesize(self.summary["memory_size"]),
                    "alert": False,
                },
            ],
            style=self.config.html.style,
        )
        mini_wordcloud = Image(
            plot_word_cloud(self.config, self.summary["plot_description"], mini=True),
            image_format=self.config.plot.image_format,
            alt="Mini wordcloud",
        )
        return Container([info, table, mini_wordcloud], sequence_type="grid")

    def _get_bottom(self):
        pass

    pass
