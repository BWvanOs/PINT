import numpy as np

def fmt1(x: float) -> str:
    """
    One decimal, no scientific notation.
    Best effort to turn a string into a number (float) and show as decimal.
    """
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return str(x)
    return f"{x:.1f}"
