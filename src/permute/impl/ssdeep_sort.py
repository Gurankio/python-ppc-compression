import threading
import typing
from pathlib import Path
from typing import Iterable
import ssdeep

from pandas import DataFrame

from permute.permuter import ParallelPermuter


class SsdeepSortPermuter(ParallelPermuter):
    """
    Sort blobs by ssdeep.
    """
    f: int
    shingles: int
    len_limit: int

    def prepare(self, df: DataFrame) -> tuple[list[tuple[int, int]], typing.Callable]:
        LSH = []
        lock = threading.Lock()

        def compute_one(index: int, path: Path, _size: int):
            lshash = ssdeep.hash(path.read_bytes())
            lock.acquire()
            LSH.append([index, lshash])
            lock.release()

        return LSH, compute_one

    def reduce(self, temp: list[tuple[int, int]], _df: DataFrame, _input_dir: Path) -> Iterable[int]:
        temp.sort(key=lambda x: x[1])
        return [item[0] for item in temp]
