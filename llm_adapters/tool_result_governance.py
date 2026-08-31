"""Deterministic byte limits for serialized tool results."""


MAX_TOOL_RESULT_BYTES = 256 * 1024
MAX_TOOL_RESULTS_PER_CYCLE_BYTES = 512 * 1024


def _validate_tool_result_bytes(
    serialized_result: str,
    cumulative_bytes: int,
) -> int:
    result_bytes = len(serialized_result.encode("utf-8"))
    if result_bytes > MAX_TOOL_RESULT_BYTES:
        raise ValueError(
            "tool result exceeds the 262144-byte context limit"
        )

    updated_bytes = cumulative_bytes + result_bytes
    if updated_bytes > MAX_TOOL_RESULTS_PER_CYCLE_BYTES:
        raise ValueError(
            "tool results exceed the 524288-byte per-cycle "
            "context limit"
        )
    return updated_bytes
