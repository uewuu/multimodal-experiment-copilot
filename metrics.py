"""通用实验指标定义。"""

from dataclasses import dataclass
from typing import Literal


MetricDirection = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class MetricSpec:
    """描述一个可配置实验指标及其展示方式。"""

    name: str
    path: tuple[str, ...]
    direction: MetricDirection
    display_name: str
    precision: int = 6

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must not be empty or whitespace")
        if self.name != self.name.strip():
            raise ValueError(
                "name must not contain leading or trailing whitespace"
            )

        if not isinstance(self.path, tuple):
            raise TypeError("path must be a tuple")
        if not self.path:
            raise ValueError("path must not be empty")
        for index, path_part in enumerate(self.path):
            if not isinstance(path_part, str):
                raise TypeError(f"path[{index}] must be a string")
            if not path_part.strip():
                raise ValueError(
                    f"path[{index}] must not be empty or whitespace"
                )
            if path_part != path_part.strip():
                raise ValueError(
                    f"path[{index}] must not contain leading or trailing whitespace"
                )

        if not isinstance(self.direction, str):
            raise TypeError("direction must be a string")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")

        if not isinstance(self.display_name, str):
            raise TypeError("display_name must be a string")
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty or whitespace")
        if self.display_name != self.display_name.strip():
            raise ValueError(
                "display_name must not contain leading or trailing whitespace"
            )

        if isinstance(self.precision, bool) or not isinstance(self.precision, int):
            raise TypeError("precision must be an integer")
        if self.precision < 0:
            raise ValueError("precision must be greater than or equal to 0")
