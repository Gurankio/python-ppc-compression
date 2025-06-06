import threading
import typing
from pathlib import Path
from typing import Iterable

import tlsh
from pandas import DataFrame

from permute.permuter import ParallelPermuter


class TlshSortPermuter(ParallelPermuter):
    """
    Sort blobs by TLSH.
    """
    f: int
    shingles: int
    len_limit: int

    def prepare(self, df: DataFrame) -> tuple[
        list[tuple[int, int]], typing.Callable[[int, Path, int], None]
    ]:
        LSH = []
        lock = threading.Lock()

        def compute_one(index: int, path: Path, _size: int):
            # https://documents.trendmicro.com/assets/wp/wp-locality-sensitive-hash.pdf
            # Byte[0] = 'T'
            # Byte[1] = '1'
            # Byte[2]Byte[3] = checksum
            # Byte[4]Byte[5] = log(length)
            # Byte[6]Byte[7] = constructed out of two 16 bit quantities derived from the quartiles: q1, q2 and q3
            lshash = tlsh.hash(path.read_bytes())[8:]

            lock.acquire()
            LSH.append([index, lshash])
            lock.release()

        return LSH, compute_one

    def reduce(self, temp: list[tuple[int, int]], _df: DataFrame, _input_dir: Path) -> Iterable[int]:
        temp.sort(key=lambda x: x[1])
        return [item[0] for item in temp]
