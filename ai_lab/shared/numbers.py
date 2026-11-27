"""Reading a whole number from what a page or a file sent.

`int(value)` takes True as 1 and 3.9 as 3, silently. A count typed as 3.9 is a
mistake to say, not a value to round, and a JSON `true` is not a number at all
(audit L35, L52). A shared file: it decides nothing, and anything may import it.
"""

from __future__ import annotations


def whole_number(value: object, what: str) -> int:
    """`value` as an int; ValueError naming `what` for a bool, a fraction or text that is not one."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a whole number, not {str(value).lower()}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{what} must be a whole number, not {value}")
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError(f"{what} must be a whole number, not {value!r}")
