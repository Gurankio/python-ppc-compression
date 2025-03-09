from pathlib import Path
from typing import Iterable

from pandas import DataFrame

from permute.permuter import Permuter


class FilenamePermuter(Permuter):
    """
    Sort blobs according to filename.
    """

    def permute(self, df: DataFrame, input_dir: Path, _num_threads: int) -> Iterable[int]:
        df['filename'] = df['filename'].str[::-1]
        permutation = df.sort_values(['filename', 'length'], ascending=[True, False]).index
        df['filename'] = df['filename'].str[::-1]
        return permutation
