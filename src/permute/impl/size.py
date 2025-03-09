from pathlib import Path
from typing import Iterable

from pandas import DataFrame

from permute.permuter import Permuter


class SizePermuter(Permuter):
    """
    Sort blobs by filename and path.
    """

    def permute(self, df: DataFrame, input_dir: Path, _num_threads: int) -> Iterable[int]:
        return df.sort_values(['length'], ascending=[False]).index
