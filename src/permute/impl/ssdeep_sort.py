import typing
from pathlib import Path
from typing import Iterable
import ssdeep

from pandas import DataFrame

from permute.permuter import ParallelPermuter


class SsdeepSortPermuter(ParallelPermuter):
    f: int
    shingles: int
    len_limit: int

    def prepare(self, df: DataFrame) -> tuple[list[tuple[int, int]], typing.Callable]:
        LSH = []

        def compute_one(index: int, path: Path, _size: int):
            lshash = ssdeep.hash(path.read_bytes())
            LSH.append([index, lshash])

        return LSH, compute_one

    def reduce(self, temp: list[tuple[int, int]]) -> Iterable[int]:
        temp.sort(key=lambda x: x[1])
        return [item[0] for item in temp]
