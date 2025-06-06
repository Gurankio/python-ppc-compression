import dataclasses
import threading
import typing
from pathlib import Path
from typing import Iterable

from pandas import DataFrame
from simhash import Simhash

from permute.permuter import ParallelPermuter
from permute.tokenizer import Tokenizer
from utils.generic import mySHA256


@dataclasses.dataclass(frozen=True)
class SimHashSortPermuter(ParallelPermuter, Tokenizer):
    """
    Sort blobs by simhash.
    """
    f: int
    shingles: int
    len_limit: int

    def prepare(self, df: DataFrame) -> tuple[list[tuple[int, int]], typing.Callable[[int, Path, int], None]]:
        lshasher = None

        if self.f == 64:
            def lshasher(features):
                import spookyhash

                return Simhash(features, hashfunc=spookyhash.hash64, f=self.f).value

        if self.f == 128:
            def lshasher(features):
                import spookyhash

                return Simhash(features, hashfunc=spookyhash.hash128, f=self.f).value

        if self.f == 256:
            def lshasher(features):
                return Simhash(features, hashfunc=mySHA256, f=self.f).value

        assert lshasher is not None

        LSH = []
        lock = threading.Lock()

        def compute_one(index: int, path: Path, size: int):
            # Check if size pf the file is < 1MiB
            if size < 2 ** 20:
                features = self.tokenize(path.read_text(errors='ignore'), self.shingles, self.len_limit)
                lshash = lshasher(features)

                lock.acquire()
                LSH.append([index, lshash])
                lock.release()
            else:
                lock.acquire()
                LSH.append([index, 0])
                lock.release()

        return LSH, compute_one

    def reduce(self, temp: list[tuple[int, int]], _df: DataFrame, _input_dir: Path) -> Iterable[int]:
        temp.sort(key=lambda x: x[1])
        return [item[0] for item in temp]
