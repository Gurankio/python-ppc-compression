from pathlib import Path
from typing import Iterable

from pandas import DataFrame

from permute.permuter import Permuter


class FilenamePathPermuter(Permuter):
    """
    Sort blobs by filename and path.
    """

    def permute(self, df: DataFrame, input_dir: Path, _num_threads: int) -> Iterable[int]:
        df['filepath'] = df['filepath'].str[::-1]
        return df.sort_values(['filepath', 'length'], ascending=[True, False]).index
