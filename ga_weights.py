"""Rank weights in mass, temperature, drying order."""
from math import isfinite


def parse_weights(text):
    try:
        weights = tuple(float(part) for part in text.split(":"))
    except ValueError:
        weights = ()
    if len(weights) != 3 or not all(isfinite(w) and w >= 0 for w in weights) or not any(weights):
        raise ValueError("Use three nonnegative weights, e.g. 1:1:3, with at least one positive.")
    return weights
