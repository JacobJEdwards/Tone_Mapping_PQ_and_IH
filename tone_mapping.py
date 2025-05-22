import numpy as np
import pq_transfer


def tone_map_pq_histogram(
    hdr_image: np.ndarray, bins: int = 256, k: int = 5
) -> np.ndarray:
    hdr_image = hdr_image.astype(np.float32)

    # algorithm 2
    L = (
        0.2126 * hdr_image[:, :, 0]
        + 0.7152 * hdr_image[:, :, 1]
        + 0.0722 * hdr_image[:, :, 2]
    )

    L_pq = pq_transfer.linear_to_pq(L)

    # algorithm 3
    min_L = np.min(L_pq)
    max_L = np.max(L_pq)

    bin_edges = np.zeros(bins + 1)
    for i in range(bins + 1):
        bin_edges[i] = min_L + i * (max_L - min_L) / bins

    hist, _ = np.histogram(L_pq, bins=bin_edges)

    # truncate histogram according to the bit after the algorithms, limiting the number of pixels in each bin
    total_pixels = L_pq.size
    uniform_bin_count = total_pixels / bins

    max_bin_count = k * uniform_bin_count
    hist = np.minimum(hist, max_bin_count)

    lut = np.zeros((bins + 1, 2))
    lut[:, 0] = bin_edges

    # algorithm 4
    lut[0, 1] = 0
    for i in range(1, bins + 1):
        lut[i, 1] = lut[i - 1, 1] + hist[i - 1]

    # normalised to [0, 255] as per algo 5
    lut[:, 1] = 255 * lut[:, 1] / lut[bins, 1]

    # algorithm 6 (pretty sure its mapped correctly)
    L_ldr = np.zeros_like(L_pq)
    for i in range(bins):
        mask = (L_pq >= lut[i, 0]) & (L_pq < lut[i + 1, 0])

        if np.any(mask):
            t = (L_pq[mask] - lut[i, 0]) / (lut[i + 1, 0] - lut[i, 0])
            L_ldr[mask] = lut[i, 1] + t * (lut[i + 1, 1] - lut[i, 1])

    L_ldr[L_pq == max_L] = 255

    result = np.zeros_like(hdr_image)
    for c in range(3):
        mask = L > 0
        if np.any(mask):
            ratio = np.zeros_like(hdr_image[:, :, c])
            ratio[mask] = hdr_image[:, :, c][mask] / L[mask]
            ratio = np.clip(ratio, 0, 3)
            result[:, :, c][mask] = ratio[mask] * L_ldr[mask]

    result = np.clip(result, 0, 255)

    return result
