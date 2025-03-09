import threading
import typing
from abc import ABC, abstractmethod
from functools import cached_property

from pathlib import Path
from typing import Iterable

from pandas import DataFrame

from permute.permuter import ParallelPermuter
from utils.generic import byte_size_list_rows, tlsh_sort_list, row_minhashgraph_unionfind_fun


# import magic


class BaseMagikaPermuter(ParallelPermuter, ABC):
    @cached_property
    def magika(self):
        from magika import Magika
        return Magika()

    def __init__(self):
        # Actually load the model.
        _ = self.magika

    def guess_from_bytes(self, x: Path):
        # 4096 is memory page size
        return self.magika.identify_bytes(open(x, "rb").read(4096)).output.ct_label

    def guess_from_path(self, x: Path):
        # TODO: To make work magika with the path.
        return self.magika.identify_path(x).output.ct_label

    @abstractmethod
    def guess(self, x: Path):
        ...

    def prepare(self, df: DataFrame) -> tuple[dict, typing.Callable[[int, Path, int], None]]:
        lock = threading.Lock()
        map_type_rows = {}

        def compute_one_file_type(index: int, path: Path, size: int):
            if size > 2 ** 20:
                file_type = 'too_big'
            elif size < 200:
                file_type = 'too_small'
            else:
                file_type = self.guess(path)

            lock.acquire()
            if file_type not in map_type_rows:
                map_type_rows[file_type] = [index]
            else:
                map_type_rows[file_type].append(index)
            lock.release()

        return map_type_rows, compute_one_file_type

    @abstractmethod
    def sort(self, df: DataFrame, row_list):
        ...

    def reduce(self, temp: dict) -> Iterable[int]:
        permutation = []

        for t, row_list in temp.items():
            if byte_size_list_rows(df, row_list) > (2 * (2 ** 20)) and len(row_list) > 3:
                # TODO: input_dir=input_dir
                tmp = self.sort(df=df, row_list=row_list)
                permutation.extend(tmp)
            else:
                tmp = sorted(row_list, key=lambda x: int(df.iloc[x]['length']), reverse=True)
                permutation.extend(tmp)

        return permutation


class MagikaPermuter(BaseMagikaPermuter):

    def guess(self, x: Path):
        return self.guess_from_bytes(x)

    def sort(self, df: DataFrame, row_list):
        # TODO: check
        return row_list


class MagikaTlshSortPermuter(BaseMagikaPermuter):

    def guess(self, x: Path):
        return self.guess_from_bytes(x)

    def sort(self, df: DataFrame, row_list):
        # TODO: check
        return tlsh_sort_list(df, row_list, input_dir)


class MagikaMinHashGraphPermuter(BaseMagikaPermuter):

    def guess(self, x: Path):
        return self.guess_from_bytes(x)

    def sort(self, df: DataFrame, row_list):
        # TODO: check
        return row_minhashgraph_unionfind_fun(df, row_list, input_dir)
