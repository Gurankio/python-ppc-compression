import dataclasses
import re
import threading
import typing
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from pandas import DataFrame
from simhash import Simhash

from permute.permuter import ParallelPermuter
from utils.generic import mySHA256


@dataclasses.dataclass(frozen=True)
class BaseMeaningfulSimHashPermuter(ParallelPermuter, ABC):
    chunk_size: int
    f: int
    shingles: int
    len_limit: int

    @abstractmethod
    def pattern(self) -> re.Pattern:
        ...

    def prepare(self, df: DataFrame) -> tuple[list[tuple[int, int]], typing.Callable[[int, Path, int], None]]:
        mapping = []
        lock = threading.Lock()
        pattern = self.pattern()

        if pattern.groups == 1:
            def joiner(matches):
                return matches
        else:
            def joiner(matches):
                return list(''.join(match) for match in matches)

        def compute_one(index, path, size):
            with open(path, mode='rt', errors='ignore') as file:
                chunk = file.read(min(size, self.chunk_size))

            # Get only the names
            matches = pattern.findall(chunk)
            names = joiner(matches)
            lsh = Simhash(names, hashfunc=mySHA256, f=self.f).value

            lock.acquire()
            mapping.append([index, lsh])
            lock.release()

        # assert (check_is_permutation(mapping_0, num_blobs))
        return mapping, compute_one

    def reduce(self, temp: list[tuple[int, int]], _df: DataFrame, _input_dir: Path) -> Iterable[int]:
        temp.sort(key=lambda x: x[1])
        return [item[0] for item in temp]


class ClassSimHashPermuter(BaseMeaningfulSimHashPermuter):
    """
    Sorts blobs by the SimHash of the class names they contain.
    """

    def pattern(self) -> re.Pattern:
        return re.compile(r'class\s+(\S+?)[:(]')


class ClassFunctionSimHashPermuter(BaseMeaningfulSimHashPermuter):
    """
    Sorts blobs by the SimHash of the class or function names they contain.
    """

    def pattern(self) -> re.Pattern:
        return re.compile(r'class\s+(\S+?)[:(]|def\s+(\S+?)\(.*?\):')


class ImportClassFunctionSimHashPermuter(BaseMeaningfulSimHashPermuter):
    """
    Sorts blobs by the SimHash of the import, class or function names they contain.
    """

    def pattern(self) -> re.Pattern:
        return re.compile(r'import\s+(\S.+?)|class\s+(\S+?)[:(]|def\s+(\S+?)\(.*?\):')
