import time
import typing
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from pandas import DataFrame


class Permuter(ABC):

    @abstractmethod
    def permute(self, df: DataFrame, input_dir: Path, num_threads: int) -> Iterable[int]:
        ...

    def permute_and_measure(self, df: DataFrame, input_dir: Path, num_threads) -> tuple[float, Iterable[int]]:
        start = time.time()
        output = self.permute(df, input_dir, num_threads)
        elapsed = time.time() - start
        return elapsed, output


class ParallelPermuter[T](Permuter):

    def permute(self, df: DataFrame, input_dir: Path, num_threads: int) -> Iterable[int]:
        temp, compute_one = self.prepare(df)

        if num_threads > 1:
            with ThreadPoolExecutor(num_threads) as executor:
                for row in range(len(df.index)):
                    path = input_dir / df.iloc[row]['local_path'] / df.iloc[row]['file_id']
                    size = int(df.iloc[row]['length'])
                    executor.submit(compute_one, row, path, size)
        else:
            for row in range(len(df.index)):
                path = input_dir / df.iloc[row]['local_path'] / df.iloc[row]['file_id']
                size = int(df.iloc[row]['length'])
                compute_one(row, path, size)

        return self.reduce(temp, df, input_dir)

    @abstractmethod
    def prepare(self, df: DataFrame) -> tuple[T, typing.Callable[[int, Path, int], None]]:
        ...

    @abstractmethod
    def reduce(self, temp: T, df: DataFrame, input_dir: Path) -> Iterable[int]:
        ...
