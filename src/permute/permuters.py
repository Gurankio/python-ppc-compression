import textwrap
from abc import ABCMeta, ABC
from enum import EnumMeta, Enum

from permute.impl.filename import FilenamePermuter
from permute.impl.filename_path import FilenamePathPermuter
from permute.permuter import Permuter
from permute.impl.random import RandomPermuter
from permute.impl.identity import IdentityPermuter
from permute.impl.meaningful_names import ClassNamesPermuter, ClassFunctionNamesPermuter, \
    ImportClassFunctionNamesPermuter
from permute.impl.minhash_graph import MinHashGraph
from permute.impl.simhash_sort import SimHashSortPermuter


class ABCEnumMeta(ABCMeta, EnumMeta):
    pass


class Permuters(Permuter, ABC, Enum, metaclass=ABCEnumMeta):
    RANDOM = RandomPermuter()
    LIST = IdentityPermuter()
    FILENAME = FilenamePermuter()
    FILENAME_PATH = FilenamePathPermuter()
    SIMHASH_SORT = SimHashSortPermuter(256, 1, 10)
    MINHASH_GRAPH = MinHashGraph(256, 64, 1, 10)
    C_NAMES = ClassNamesPermuter(2 ** 16)
    CF_NAMES = ClassFunctionNamesPermuter(2 ** 16)
    ICF_NAMES = ImportClassFunctionNamesPermuter(2 ** 16)

    @classmethod
    def choices(cls) -> list[str]:
        return [p.name.lower() for p in cls]

    @classmethod
    def help(cls) -> str:
        return 'Permutation strategies, one or more of the following:\n' + '\n'.join(
            f'* {p.name.lower()}: {textwrap.dedent(p.value.__class__.__doc__).strip()}'
            for p in cls
            if p.value.__class__.__doc__ is not None
        )
