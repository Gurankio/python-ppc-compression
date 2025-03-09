from pathlib import Path

import numpy as np
from pandas import DataFrame

from permute.permuter import Permuter


class RandomPermuter(Permuter):
    """
    Permute blobs randomly.
    """

    def permute(self, df: DataFrame, input_dir: Path, _num_threads: int):
        np.random.seed(42)
        return np.random.permutation(len(df.index))
