import threading
from functools import cached_property
from typing import Any, Callable, Iterable

from pandas import DataFrame

from permute.permuter import ParallelPermuter
from utils.generic import byte_size_list_rows, row_minhashgraph_unionfind_fun
import magic

class TypeMinHashGraphPermuter(ParallelPermuter):
    @cached_property
    def guess(self):
        from guesslang import Guess
        return Guess()

    def _guess_fun_from_content(self, x):
        return magic.from_buffer(open(x, "rb").read(2048), mime=True)

    def _guesslang(self, x):
        # read just 10K of the files
        # checks on the size are made before
        file_content = read_file_size(x, 10 * (2 ** 10))
        # assert(len(file_content) > 0)
        # file with just \n, \t, or white spaces
        if not file_content.strip():
            return 'too_small'
        return self.guess.language_name(file_content)

    def _guess_fun_guesslang_content(self, file_content):
        if not file_content.strip():
            return 'too_small'
        return self.guess.language_name(self, file_content)

    def _from_header(self, x):
        return magic.from_file(x, mime=True)

    def _guess_fun_from_header_content(self, x):
        return magic.from_buffer(x, mime=True)

    def prepare(self, _df: DataFrame) -> tuple[dict[Any, Any], Callable]:
        lock = threading.Lock()

        # TODO: refine mapping with https://guesslang.readthedocs.io/en/latest/
        map_type_rows = {}

        def compute_one_file_type(index, file_path, file_size):
            if file_size > 2 ** 20:
                file_type = 'too_big'
            elif file_size < 200:
                file_type = 'too_small'
            else:
                file_type = self._from_header(file_path)
                if 'text' in file_type:
                    file_type = self._guesslang(file_path)

            lock.acquire()
            if file_type not in map_type_rows:
                map_type_rows[file_type] = [index]
            else:
                map_type_rows[file_type].append(index)
            lock.release()

        return (map_type_rows, df), compute_one_file_type

    def reduce(self, temp: dict) -> Iterable[int]:
        map_type_rows, df = temp
        permutation = []

        for t, row_list in map_type_rows.items():
            if (byte_size_list_rows(df, row_list) > (2 * (2 ** 20))) and len(row_list) > 3:
                tmp = row_minhashgraph_unionfind_fun(df=df, row_list=row_list, input_dir=input_dir)
                permutation.extend(tmp)
            else:
                tmp = sorted(row_list, key=lambda x: int(df.iloc[x]['length']), reverse=True)
                permutation.extend(tmp)

        return permutation
