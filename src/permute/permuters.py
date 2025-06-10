import textwrap
from abc import ABCMeta, ABC
from enum import EnumMeta, Enum

from permute.impl.embed_meaningful_names import ClassFunctionNameEmbedPermuter
from permute.impl.filename import FilenamePermuter
from permute.impl.filename_path import FilenamePathPermuter
from permute.impl.first_meaningful_name import FirstClassFunctionNamePermuter
from permute.impl.graph_meaningful_names import ClassFunctionNameGraphPermuter
from permute.impl.meaningful_simhash import ClassFunctionSimHashPermuter, ImportClassFunctionSimHashPermuter, \
    ClassSimHashPermuter
from permute.permuter import Permuter
from permute.impl.random import RandomPermuter
from permute.impl.identity import IdentityPermuter
from permute.impl.meaningful_names import ClassFunctionNamesPermuter
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

    # C_NAMES = ClassNamesPermuter(2 ** 16)
    CF_NAMES = ClassFunctionNamesPermuter(2 ** 16)
    # ICF_NAMES = ImportClassFunctionNamesPermuter(2 ** 16)

    FIRST_CF_NAME = FirstClassFunctionNamePermuter(2 ** 12)

    CF_NAME_GRAPH = ClassFunctionNameGraphPermuter(2 ** 16)

    CF_NAMES_EMBED = ClassFunctionNameEmbedPermuter(2 ** 16)

    C_SIMHASH_SORT = ClassSimHashPermuter(2 ** 16, 256, 1, 10)
    CF_SIMHASH_SORT = ClassFunctionSimHashPermuter(2 ** 16, 256, 1, 10)
    ICF_SIMHASH_SORT = ImportClassFunctionSimHashPermuter(2 ** 16, 256, 1, 10)

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
