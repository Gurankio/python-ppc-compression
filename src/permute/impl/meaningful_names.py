import dataclasses
import re
import threading
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
        lock = threading.Lock()
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

            lock.acquire()
            mapping.append([index, names])
            lock.release()

        # assert (check_is_permutation(mapping_0, num_blobs))
        return mapping, compute_one

    def reduce(self, temp: list[tuple[int, int]], _df: DataFrame, _input_dir: Path) -> Iterable[int]:
        temp.sort(key=lambda x: x[1])
        return [item[0] for item in temp]


# class ClassNamesPermuter(BaseMeaningfulNamesPermuter):
#     """
#     Sorts blobs by the class names they contain.
#     """
#
#     def pattern(self) -> re.Pattern:
#         return re.compile(rb'class\s+(\S+?)[:(]')


class ClassFunctionNamesPermuter(BaseMeaningfulNamesPermuter):
    """
    Sorts blobs by class or function names they contain.
    """

    def pattern(self) -> re.Pattern:
        return re.compile(
            rb'class\s+(\S+?):|'
            rb'def\s+(\S+?)\(.*?\):|'
            rb'class\s+(\S+?)\s*(?:extends.+?)?\s*{|'
            rb'function\s+(\S+?)\(.*?\)\s+{|'
            rb'public\s+(?:final\s+)?class\s+(\S+?)\s*(?:extends.+?|implements.+?)?\s*{|'
            rb'public\s+class\s+(\S+?)\s+{.*?(?:public|private):|'
            rb'#+\s*(.+?)|'
            rb'<title>(.+?)</title>|'
            rb'(\".+?\":)\s*[\"\[{]|'
            rb'@file\s+(.+?).c|'  # doc comment
            rb'struct\s+(\S+?)\s+{|'
            rb'export\s+const\s+(\S.+?)\s*[:=]|'
            rb'xmlns(?::.+?)?=\"(.+?)\"|'
            rb'(?:var|const)\s+(\S+?)\s*=\s*function|'
            rb'\.(\S+?)\s*{|'
            rb'<\?php.+?(?:class|interface)\s+(\S+?)',
            re.DOTALL
        )


# class ImportClassFunctionNamesPermuter(BaseMeaningfulNamesPermuter):
#     """
#     Sorts blobs by import, class or function names they contain.
#     """
#
#     def pattern(self) -> re.Pattern:
#         return re.compile(rb'import\s+(\S.+?)|class\s+(\S+?)[:(]|def\s+(\S+?)\(.*?\):')
