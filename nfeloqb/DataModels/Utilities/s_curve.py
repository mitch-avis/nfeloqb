"""Define the S-curve helper used for smooth progression and regression adjustments."""


def s_curve(height: float, mp: float, x: float, direction: str = "down") -> float:
    """Calculate an S-curve for discounting or ramping values."""
    if direction == "down":
        return (1 - (1 / (1 + 1.5 ** ((-1 * (x - mp)) * (10 / mp))))) * height
    else:
        return (1 - (1 - (1 / (1 + 1.5 ** ((-1 * (x - mp)) * (10 / mp)))))) * height
