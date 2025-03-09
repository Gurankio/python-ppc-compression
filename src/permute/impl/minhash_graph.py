import dataclasses
import typing
from pathlib import Path
from typing import Iterable

from datasketch import MinHash
from pandas import DataFrame

from permute.permuter import ParallelPermuter
from permute.tokenizer import Tokenizer
from utils.generic import byte_size_list_rows, tlsh_sort_list
from utils.unionfind import UnionFind


@dataclasses.dataclass(frozen=True)
class MinHashGraph(ParallelPermuter, Tokenizer):
    """

    """
    f: int
    r: int
    shingles: int
    len_limit: int

    def prepare(self, df: DataFrame) -> tuple[..., typing.Callable[[int, Path, int], None]]:
        LSH_tuple = []

        # TODO: what if r doesn't divide f
        #       just dont consider the rest
        b = self.f // self.r

        def add_tuple_one_file(index: int, path: Path, size: int):
            if size > 2 ** 20:
                symbol = 0

                curr_tuple = [index]
                for _ in range(self.r):
                    curr_band = []
                    for _ in range(b):
                        curr_band.append([symbol])
                    curr_tuple.append(curr_band)

                LSH_tuple.append(curr_tuple)

            else:
                m1 = MinHash(num_perm=self.f)
                for d in self.tokenize(path.read_text(errors='ignore'), self.shingles, self.len_limit):
                    m1.update(d.encode('utf8'))

                curr_tuple = [index]
                idx = 0
                for _ in range(self.r):
                    curr_band = []
                    for _ in range(b):
                        curr_band.append(m1.hashvalues[idx])
                        idx += 1

                    curr_tuple.append(curr_band)

                LSH_tuple.append(curr_tuple)

        return (len(df.index), df, LSH_tuple), add_tuple_one_file

    def reduce(self, temp) -> Iterable[int]:
        num_blobs, df, LSH_tuple = temp

        uf = UnionFind(range(num_blobs))

        # each list of Minhash is divided into r groups of b integer each

        # for each group
        for i in range(1, len(LSH_tuple[0])):
            # sort by the group
            LSH_tuple.sort(key=lambda x: x[i])
            # so that we can group together the ones that are equal
            for j in range(len(LSH_tuple) - 1):
                if LSH_tuple[j][i] == LSH_tuple[j + 1][i]:
                    uf.union(LSH_tuple[j][0], LSH_tuple[j + 1][0])

        row_list = []

        # Do this in parallel? Nope, it uses 1/100 of the time needed by add_tuple_one_file
        for connected_component in uf.components():
            list_connected_component = list(connected_component)
            if (byte_size_list_rows(df, list_connected_component) > 32 * (2 ** 20)
                    and len(list_connected_component) > 5):
                sorted_row_list = tlsh_sort_list(df, connected_component, input_dir=input_dir)
                row_list.extend(sorted_row_list)
            else:
                row_list.extend(sorted(list_connected_component,
                                       key=lambda x: int(df.iloc[x]['length']), reverse=True))

        # print(f'num_connected_components {len(uf.components())} num_blobs {num_blobs}')
        # assert (check_is_permutation(row_list, num_blobs))
        return row_list
