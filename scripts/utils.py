import numpy as np


def normalize(now: np.ndarray, low, high) -> np.ndarray:
    mean = (high + low) * 0.5
    halfwidth = (high - low) * 0.5
    normalized: np.ndarray = (now.astype(np.float32) - mean) / halfwidth
    return normalized.astype(np.float32)


def yes_no_input():
    while True:
        choice = input("Are you sure you want to continue running this code? [y/N]: ").lower()
        if choice in ["y", "ye", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False
