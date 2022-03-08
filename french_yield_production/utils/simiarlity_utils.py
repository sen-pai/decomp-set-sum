import numpy as np
from scipy.stats import wasserstein_distance

def wasserstein_severity(array_a, array_b):
    dist = wasserstein_distance(array_a, array_b)
    return dist_to_value(dist)


def dist_to_value(dist):
    if dist < 1:
        return 1 
    elif dist < 2:
        return 0.8
    elif dist < 3:
        return 0.6
    elif dist < 5:
        return 0.4
    elif dist < 7:
        return 0.2 
    elif dist < 8:
        return 0
    elif dist < 10:
        return -0.2 
    elif dist < 12:
        return -0.5
    elif dist < 14:
        return -0.7
    elif dist >= 14:
        return -1
