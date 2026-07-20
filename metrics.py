"""通用实验指标定义。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


MetricDirection = Literal["maximize", "minimize"]


def evaluate_metric_history(
    records: Sequence[Sequence[object]],
    direction: MetricDirection,
) -> dict[str, int | float]:
    """校验并评估指标历史记录。"""
    if (
        isinstance(records, (str, bytes, bytearray, Mapping))
        or not isinstance(records, Sequence)
    ):
        raise TypeError("records must be a sequence")
    if not records:
        raise ValueError("records must not be empty")

    if not isinstance(direction, str):
        raise TypeError("direction must be a string")
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be 'maximize' or 'minimize'")

    first_epoch = 0
    first_value: int | float = 0
    last_epoch = 0
    last_value: int | float = 0
    best_epoch = 0
    best_value: int | float = 0

    for index, record in enumerate(records):
        if (
            isinstance(record, (str, bytes, bytearray, Mapping))
            or not isinstance(record, Sequence)
        ):
            raise TypeError(f"records[{index}] must be a sequence")
        if len(record) != 2:
            raise ValueError(
                f"records[{index}] must contain exactly two items"
            )

        epoch, value = record
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError(f"records[{index}][0] must be an integer")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"records[{index}][1] must be a number")

        if index == 0:
            first_epoch = epoch
            first_value = value
            best_epoch = epoch
            best_value = value

        last_epoch = epoch
        last_value = value
        if (
            direction == "maximize" and value > best_value
        ) or (
            direction == "minimize" and value < best_value
        ):
            best_epoch = epoch
            best_value = value

    return {
        "record_count": len(records),
        "first_epoch": first_epoch,
        "first_value": first_value,
        "last_epoch": last_epoch,
        "last_value": last_value,
        "best_epoch": best_epoch,
        "best_value": best_value,
    }


def get_value_at_path(
    data: Mapping[str, object],
    path: tuple[str, ...],
) -> object:
    """按字符串键路径读取嵌套映射中的值。"""
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")

    if not isinstance(path, tuple):
        raise TypeError("path must be a tuple")
    if not path:
        raise ValueError("path must not be empty")
    for index, path_part in enumerate(path):
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

    current: object = data
    for index, key in enumerate(path):
        if not isinstance(current, Mapping):
            parent_path = ".".join(path[:index])
            raise TypeError(
                f"value at path '{parent_path}' must be a mapping"
            )
        if key not in current:
            current_path = ".".join(path[: index + 1])
            raise KeyError(f"missing key at path '{current_path}'")
        current = current[key]

    return current


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
