from dataclasses import dataclass

import numpy as np


@dataclass
class FrameEmbeddings:
    embeddings: np.ndarray
    frame_indices: list[int]
