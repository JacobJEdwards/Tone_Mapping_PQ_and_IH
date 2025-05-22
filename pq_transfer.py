from colour import models
import numpy as np


def pq_to_linear(pq: np.ndarray) -> np.ndarray:
    pq = pq.astype(np.float32)

    linear = models.eotf_ST2084(pq)

    return linear


def linear_to_pq(linear: np.ndarray) -> np.ndarray:
    linear = linear.astype(np.float32)

    pq = models.eotf_inverse_ST2084(linear)

    return pq

