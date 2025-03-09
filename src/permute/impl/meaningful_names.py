import dataclasses
import re
import typing
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from pandas import DataFrame

from permute.permuter import ParallelPermuter


@dataclasses.dataclass(frozen=True)
class BaseMeaningfulNamesPermuter(ParallelPermuter, ABC):
    chunk_size: int

    @abstractmethod
    def pattern(self) -> re.Pattern:
        ...

    def prepare(self, df: DataFrame) -> tuple[list[tuple[int, int]], typing.Callable[[int, Path, int], None]]:
        mapping = []
        pattern = self.pattern()

        if pattern.groups == 1:
            def joiner(matches):
                return tuple(matches)
        else:
            def joiner(matches):
                return tuple(b''.join(match) for match in matches)

        def compute_one(index, path, size):
            with open(path, mode='rb') as file:
                chunk = file.read(min(size, self.chunk_size))

            # Get only the names
            matches = pattern.findall(chunk)
            names = joiner(matches)

            mapping.append([index, names])

        # assert (check_is_permutation(mapping_0, num_blobs))
        return mapping, compute_one

    def reduce(self, temp: list[tuple[int, int]]) -> Iterable[int]:
        temp.sort(key=lambda x: x[1])
        return [item[0] for item in temp]


class ClassNamesPermuter(BaseMeaningfulNamesPermuter):
    def pattern(self) -> re.Pattern:
        return re.compile(rb'class\s+(\S+?):')


class ClassFunctionNamesPermuter(BaseMeaningfulNamesPermuter):
    def pattern(self) -> re.Pattern:
        return re.compile(rb'class\s+(\S+?):|def\s+(\S+?)\(.*?\):')


class ImportClassFunctionNamesPermuter(BaseMeaningfulNamesPermuter):
    def pattern(self) -> re.Pattern:
        return re.compile(rb'import\s+(\S.+?)|class\s+(\S+?):|def\s+(\S+?)\(.*?\):')
