import dataclasses
import re
import threading
import typing
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import tlsh
from pandas import DataFrame

from permute.permuter import ParallelPermuter


@dataclasses.dataclass(frozen=True)
class BaseFirstMeaningfulNamePermuter(ParallelPermuter, ABC):
    chunk_size: int

    @abstractmethod
    def pattern(self) -> re.Pattern:
        ...

    @abstractmethod
    def language_index(self, index: int) -> str:
        ...

    def prepare(self, df: DataFrame) -> tuple[list[tuple[int, int]], typing.Callable[[int, Path, int], None]]:
        mapping = []
        lock = threading.Lock()
        pattern = self.pattern()

        def compute_one(index, path, size):
            with open(path, mode='rb') as file:
                chunk = file.read(min(size, self.chunk_size))

            # Get only the names
            match = next(map(lambda m: m.groups(), pattern.finditer(chunk)), tuple())
            match = next(((self.language_index(i), x) for i, x in enumerate(match) if x is not None), None)

            if match is None:
                # print(f'{df.iloc[index]['filename']:30}',
                #       df.iloc[index]['local_path'] + '/' + df.iloc[index]['file_id'])
                match = ('zzz_tlsh', tlsh.hash(chunk)[8:])

            lock.acquire()
            mapping.append([index, match])
            lock.release()

        # assert (check_is_permutation(mapping_0, num_blobs))
        return mapping, compute_one

    def reduce(self, temp: list[tuple[int, int]], _df: DataFrame, _input_dir: Path) -> Iterable[int]:
        temp.sort(key=lambda x: x[1])
        return [item[0] for item in temp]


class FirstClassFunctionNamePermuter(BaseFirstMeaningfulNamePermuter):
    """
    Sorts blobs by class or function names they contain.
    """

    LANGUAGES = {
        0: 'py',
        1: 'py',
        2: 'js',
        3: 'js',
        4: 'java',
        5: 'cpp',
        6: 'md',
        7: 'html',
        8: 'json',
        9: 'c',
        10: 'c',
        11: 'ts',
        12: 'xml',
        13: 'js',
        14: 'css',
        15: 'php',
    }

    def language_index(self, index: int) -> str:
        return self.LANGUAGES[index]

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
