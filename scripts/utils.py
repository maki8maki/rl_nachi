import numpy as np

def normalize(now: np.ndarray, low, high) -> np.ndarray:
    mean = (high + low) * 0.5
    halfwidth = (high - low) * 0.5
    normalized: np.ndarray = (now.astype(np.float32) - mean) / halfwidth
    return normalized.astype(np.float32)
