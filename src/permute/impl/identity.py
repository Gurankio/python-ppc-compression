from pathlib import Path
from typing import Iterable

from pandas import DataFrame

from permute.permuter import Permuter


class IdentityPermuter(Permuter):
    """
    No permutation, just use the order in the csv list.
    """

    def permute(self, df: DataFrame, input_dir: Path, _num_threads: int) -> Iterable[int]:
        # just take the order in which the blobs are listed in the dataframe
        return range(len(df.index))
