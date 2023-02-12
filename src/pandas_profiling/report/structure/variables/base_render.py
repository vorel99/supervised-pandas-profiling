from abc import ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Any, Dict

from pandas_profiling.config import Settings
from pandas_profiling.report.presentation.core.variable_info import VariableInfo
from pandas_profiling.report.structure.variables import render_common


@dataclass
class BaseRenderVariable(metaclass=ABCMeta):
    config: Settings
    summary: Dict[str, Any]

    def _get_info(self) -> VariableInfo:
        """Return rendered basic info about variable."""
        return VariableInfo(
            anchor_id=self.summary["varid"],
            alerts=self.summary["alerts"],
            var_type=self.summary["type"],
            var_name=self.name,
            description=self.summary["description"],
            style=self.config.html.style,
        )

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def _get_top(self):
        """Return top section of rendered variable."""
        pass

    @abstractmethod
    def _get_bottom(self):
        """Return bottom section of rendered variable."""
        pass

    def render(self):
        """Return template for variable prot."""
        template_variables = render_common(self.config, self.summary)
        template_variables["top"] = self._get_top()
        template_variables["bottom"] = self._get_bottom()
        return template_variables

    pass
