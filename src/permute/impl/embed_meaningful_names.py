import dataclasses
import re
import threading
import typing
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from time import time
from typing import Iterable, Any, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy
import tlsh
from pandas import DataFrame

from permute.permuter import ParallelPermuter
from utils.unionfind import UnionFind


@dataclasses.dataclass(frozen=True)
class BaseMeaningfulNameEmbedPermuter(ParallelPermuter, ABC):
    chunk_size: int

    @abstractmethod
    def pattern(self) -> re.Pattern:
        ...

    @abstractmethod
    def language_index(self, index: int) -> str:
        ...

    def prepare(self, df: DataFrame) -> tuple[dict[Any, Any], Callable[[Any, Any, Any], None]]:
        mapping = {}
        lock = threading.Lock()
        pattern = self.pattern()

        def joiner(matches):
            def inner():
                for match in matches:
                    yield from ((self.language_index(i), x) for i, x in enumerate(match) if x is not None)

            return tuple(set(inner()))

        def compute_one(index, path, size):
            with open(path, mode='rb') as file:
                chunk = file.read(min(size, self.chunk_size))

            # Get only the names
            matches = pattern.findall(chunk)
            names = joiner(matches)

            lock.acquire()
            mapping[index] = names
            lock.release()

        # assert (check_is_permutation(mapping_0, num_blobs))
        return mapping, compute_one

    def reduce(self, temp: dict[int, list[bytes]], _df: DataFrame, _input_dir: Path) -> Iterable[int]:
        print(f'  Reduce              @ {time():.3f}')

        # Build the inverse list: term -> files
        inverse = {}
        for file, terms in temp.items():
            for term in terms:
                inverse.setdefault(term, set())
                inverse[term].add(file)

        print(f'  Inverted            @ {time():.3f}')
        print(f'  (assumption: #terms >> #files -> {len(inverse)} >> {len(temp)})')

        inverse_sorted = list(sorted(inverse.values(), key=len, reverse=True)[:8192])
        print(f'  Sorted terms        @ {time():.3f}')

        embedding = {}
        lock = threading.Lock()

        def make_embed(f, ts):
            e = tuple(t in ts for t in inverse_sorted)
            with lock:
                embedding[f] = e

        with ThreadPoolExecutor(16) as executor:
            for file, terms in temp.items():
                executor.submit(make_embed, file, terms)
        print(f'  Built file embeddings @ {time():.3f}')

        output = sorted(temp.keys(), key=lambda f: embedding[f], reverse=True)
        print(f'  Sorted files          @ {time():.3f}')

        return output

    # v2: high freq
    # def reduce(self, temp: dict[int, list[bytes]], _df: DataFrame, _input_dir: Path) -> Iterable[int]:
    #     print(f'Reduce @ {time()}')
    #
    #     # Build the inverse list: term -> files
    #     inverse = {}
    #     for file, terms in temp.items():
    #         for term in terms:
    #             inverse.setdefault(term, set())
    #             inverse[term].add(file)
    #
    #     inverse_sorted = list(sorted(inverse.values(), key=len, reverse=True)[:4096])
    #     print(f'  Inverted and Sorted @ {time()}')
    #
    #     embedding = {}
    #     lock = threading.Lock()
    #
    #     def compute_one(f):
    #         e = tuple(f in fs for fs in inverse_sorted)
    #         with lock:
    #             embedding[f] = e
    #
    #     with ThreadPoolExecutor(16) as executor:
    #         for f in temp.keys():
    #             executor.submit(compute_one, f)
    #     print(f'  Built embeddings @ {time()}')
    #
    #     output = list(sorted(temp.keys(), key=lambda f: embedding[f]))
    #     print(f'  Sorted @ {time()}')
    #     return output

    # v1: low freq, with UnionFind not reported, but did terribly.


# class ClassNameGraphPermuter(BaseMeaningfulNameGraphPermuter):
#     """
#     Sorts blobs by the class names they contain.
#     """
#
#     def pattern(self) -> re.Pattern:
#         return re.compile(rb'class\s+(\S+?):')


class ClassFunctionNameEmbedPermuter(BaseMeaningfulNameEmbedPermuter):
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

# class ImportClassFunctionNameGraphPermuter(BaseMeaningfulNameGraphPermuter):
#     """
#     Sorts blobs by import, class or function names they contain.
#     """
#
#     def pattern(self) -> re.Pattern:
#         return re.compile(rb'import\s+(\S.+?)|class\s+(\S+?):|def\s+(\S+?)\(.*?\):')
