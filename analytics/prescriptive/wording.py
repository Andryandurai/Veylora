"""Plain-language wording of a paired difference and its 95 percent interval (shared by the report and the dashboards).

The verdict is computed from the numbers each time, so a re-run that changes an interval also changes the sentence.
"""

from __future__ import annotations

SMALL = 0.05  # differences in relative yield below this are called small


def paired_phrase(
    delta: float,
    lo: float,
    hi: float,
    gain: str = "gain",
    loss: str = "loss",
    none: str = "no clear difference",
) -> str:
    """Words for a difference with its interval.

    ``none`` when the interval contains zero; otherwise ``a clear gain`` or ``a clear loss``, written ``a clear but
    small ...`` when the difference is under ``SMALL`` so that a barely-excluded zero is not oversold.
    """
    if lo <= 0.0 <= hi:
        return none
    word = gain if delta > 0 else loss
    return f"a clear but small {word}" if abs(delta) < SMALL else f"a clear {word}"
