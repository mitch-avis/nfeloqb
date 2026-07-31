"""Define the progressive discount helper used to tame large game-value errors."""


def prog_disc(obs: float, proj: float, scale: float, alpha: float) -> float:
    """Progressively discount large deviations from the projected value."""
    # calculate the error
    abs_error = abs(obs - proj)
    error_direction = 1 if obs >= proj else -1
    # control for instances with no error or discounting
    if abs_error == 0 or alpha == 0:
        return obs
    # attempt to calc processed value while controlling for overflow errors
    try:
        return proj + (
            error_direction
            *
            # process error
            min(abs_error, 0.309 * (alpha**-0.864) * scale)
            ** (1 - min((min(abs_error, 0.309 * (alpha**-0.864) * scale) / scale) * alpha, 1))
        )
    except OverflowError:
        return obs
