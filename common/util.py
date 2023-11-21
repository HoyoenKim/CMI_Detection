import math
import os
import random
import sys
import time

import numpy as np
import psutil

def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)

def pad_if_needed(x, max_len, pad_value = 0.0):
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)

def nearest_valid_size(input_size, downsample_rate):
    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size

def random_crop(pos, duration, max_end):
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


def negative_sampling(this_event_df, num_steps):
    positive_positions = set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]

# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length, sigma = 3):
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma)) 
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_label(label, offset, sigma):
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

    return label
