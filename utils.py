import itertools
import re
import subprocess
import os
import numpy as np
from simhash import Simhash
import spookyhash
from datasketch import MinHash
import tlsh
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import Counter
from matplotlib import pyplot as plt
import git
import time
import sys

# check if python version is 3.8 or higher
if sys.version_info < (3, 9):
    import ssdeep

is_MinHash = False

start_block_size_exp = 21
end_block_size_exp = 25

start_clusters_div_exp = 26
end_clusters_div_exp = 27

float_1GiB = float(2**30)
float_1MiB = float(2**20)
float_1KiB = float(2**10)

ones_8 = 2**8 - 1
ones_16 = 2**16 - 1
ones_32 = 2**32 - 1
ones_64 = 2**64 - 1

len_limit = 10

STEP_PRINT = 100000

BASE_WORKING_DIR = '/extralocal/swh'

COMPRESSED_BLOB_DIR = '/extralocal/swh/blobs_compressed'

NUM_THREAD = 16

REPO_DIR = os.path.dirname(os.path.realpath(__file__))

repo = git.Repo(search_parent_directories=True)


def gray_code(x):
    return x ^ (x >> 1)


def row_sorting_fun(df, row_list): return simhash_sort_list(
    df, row_list, 1, 128, len_limit)


def row_simhashgraph128_unionfind_fun(df, row_list): return simhash_graph_unionfind_list(
    df, row_list, 1, 128, 32, len_limit)


def row_tlshsort_fun(df, row_list): return tlsh_sort_list(
    df, row_list)


def row_simhashgraph256_unionfind_fun(df, row_list): return simhash_graph_unionfind_list(
    df, row_list, 1, 256, 32, len_limit)


def row_minhashgraph_unionfind_fun(df, row_list, input_dir): return minhash_graph_unionfind_list(
    df, row_list, 1, 256, 64, len_limit, input_dir)


def exec_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    process.wait()
    output, error = process.communicate()
    if error != '':
        return output
    else:
        print('ERROR --> ' + error)
        return output


def guess_fun_from_linguistic(x): return json.loads(
    exec_cmd('github-linguist --json {}'.format(x)))[x]['language']


def mySHA256(x):
    m_sha256 = hashlib.sha256()
    m_sha256.update(x)
    return m_sha256.digest()


def get_fixed_num_features_content(s):
    width = len(s) - 100 if len(s) > 100 else len(s)
    ngrams = [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
    return ngrams


def get_tokens_content(s, width=3):
    ngrams = [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
    return ngrams


def get_tokens(file_content, width=1, len_limit=10):
    # get a list of lines from the content of the file
    tokens = file_content.split('\n')
    if width > 1:
        # tokes are consecutive lines grouped togheter
        tokens = [tokens[i:i+width][0]
                  for i in range(max(len(tokens)-width+1, 1))]
    # remove lines with less than 10 chars and delete leading and trailing tabs and whitespaces
    tokens = [x.strip() for x in tokens if len(x) > len_limit]
    return tokens


def get_tokens_bytes(file_content, width=1, len_limit=10):
    # get a list of lines from the content of the file
    tokens = file_content.split(b'\n')
    if width > 1:
        # tokes are consecutive lines grouped togheter
        tokens = [tokens[i:i+width][0]
                  for i in range(max(len(tokens)-width+1, 1))]
    # remove lines with less than 10 chars and delete leading and trailing tabs and whitespaces
    # it ust return a list of str
    tokens = [str(x.strip()) for x in tokens if len(x) > len_limit]
    return tokens


def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def check_is_permutation(arr, n):
    # Set to check the count
    # of non-repeating elements
    s = set()
    maxEle = 0
    if len(arr) != n:
        print('len(arr) = {} != n = {}'.format(len(arr), n))
        print(Diff(arr, range(n)))
        return False

    for i in range(n):
        # Insert all elements in the set
        s.add(arr[i])
        # Calculating the max element
        maxEle = max(maxEle, arr[i])
    if (maxEle != n-1):
        print("max elem not equal n-1")
        print(Diff(arr, range(n)))
        return False
    # Check if set size is equal to n
    if (len(s) == n):
        return True

    print("the array contains duplicates")
    print(Diff(arr, range(n)))
    return False


def check_is_permutation_list(arr1, arr2):
    if len(arr1) != len(arr2):
        print("len(arr1) = {} != len(arr2) = {}".format(len(arr1), len(arr2)))
        print(Diff(arr1, arr2))
        return False
    res = Counter(arr1) == Counter(arr2)
    if not res:
        print("the arrays are not permutations")
        print(Diff(arr1, arr2))
        return False
    else:
        return True


def read_file(filename):
    with open(filename, mode='r', encoding="utf-8", errors="ignore") as file:
        all_of_it = file.read()
        return all_of_it


def read_file_get_minhash_on_tokens(filename, width, len_limit, f):
    with open(filename, mode='r', encoding="utf-8", errors="ignore") as file:
        all_of_it = file.read()
        tokens = all_of_it.split('\n')
        if width > 1:
            # tokes are consecutive lines grouped togheter
            tokens = [tokens[i:i+width][0]
                      for i in range(max(len(tokens)-width+1, 1))]
        # remove lines with less than 10 chars and delete leading and trailing tabs and whitespaces
        tokens = [x.strip() for x in tokens if len(x) > len_limit]
        m1 = MinHash(num_perm=f)
        for d in tokens:
            m1.update(d.encode('utf8'))

        return m1


def read_file_size(filename, size):
    with open(filename, mode='r', encoding="utf-8", errors="ignore") as file:
        all_of_it = file.read(size)
        return all_of_it


def read_file_bytes(filename):
    with open(filename, mode='rb') as file:
        all_of_it = file.read()
        return all_of_it


def write_filenames_to_file(list_to_write, filename):
    with open(filename, mode='w') as file:
        file.write('\n'.join(list_to_write))


def du(path):
    """disk usage in human readable format, in bytes"""
    # print('Calling --> du ' + path)
    return subprocess.check_output(['du', '-sb', path]).split()[0].decode('utf-8')


def str_block_size(str_block_size):
    return 'block_size=' + str_block_size

def byte_size_list_rows(df, row_list):
    to_return = 0
    for row in row_list:
        to_return += df.iloc[row]['length']

    return to_return


def simhash_sort_list(df, row_list, shingle_num, _f, len_limit, input_dir):
    LSH = []
    for index, row in enumerate(row_list):
        path_file = os.path.join(
            input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        len_file = int(df.iloc[row]['length'])
        lshash = 0
        if len_file < 2**20:
            # read all lines at once
            all_of_it = read_file(path_file)
            features = get_tokens(all_of_it, shingle_num, len_limit)
            Simhash(features, hashfunc=spookyhash.hash128, f=_f).value
        LSH.append([row, lshash])

    LSH.sort(key=lambda x: x[1])
    LSH_0 = [item[0] for item in LSH]
    # assert (check_is_permutation_list(LSH_0, row_list))
    return LSH_0


def tlsh_sort_list(df, row_list, input_dir):
    LSH = []
    for index, row in enumerate(row_list):
        path_file = os.path.join(
            input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        file_size = df.iloc[row]['length']
        lshash = '0'
        # TLSH is fast so limit at 4MiB
        if file_size < 2**22:
            # read all lines at once
            all_of_it = read_file_bytes(path_file)
            lshash = tlsh.hash(all_of_it)[8:]
        LSH.append([row, lshash])

    LSH.sort(key=lambda x: x[1])
    LSH_0 = [item[0] for item in LSH]
    # assert (check_is_permutation_list(LSH_0, row_list))
    return LSH_0


def simhash_graph_unionfind_list(df, row_list, shingle_num, f, r, len_limit, input_dir):
    num_blobs = len(row_list)

    LSH_tuple = []

    # b = f // r

    total_uncompressed_size = df.iloc[row_list]['length'].sum()
    # the list could contain few (big) files so
    if num_blobs < 3 or total_uncompressed_size < 32 * (2 ** 20):  # 32MiB
        return sorted(row_list, key=lambda x: int(df.iloc[x]['length']), reverse=True)

    LSH_tuple = []

    if f == 128:
        def add_tuple_one_file(path_file, index):
            all_of_it = read_file(path_file)
            features = get_tokens(all_of_it, shingle_num, len_limit)
            lshash = Simhash(
                features, hashfunc=spookyhash.hash128, f=f).value
            if r == 4:
                LSH_tuple.append([index,
                                  lshash & ones_32,
                                  (lshash >> 32) & ones_32,
                                  (lshash >> 64) & ones_32,
                                  (lshash >> 96) & ones_32])
            elif r == 8:
                LSH_tuple.append([index,
                                  lshash & ones_16,
                                  (lshash >> 16) & ones_16,
                                  (lshash >> 32) & ones_16,
                                  (lshash >> 48) & ones_16,
                                  (lshash >> 64) & ones_16,
                                  (lshash >> 80) & ones_16,
                                  (lshash >> 96) & ones_16,
                                  (lshash >> 112) & ones_16])
            elif r == 16:
                LSH_tuple.append([index,
                                  lshash & ones_8,
                                  (lshash >> 8) & ones_8,
                                  (lshash >> 16) & ones_8,
                                  (lshash >> 24) & ones_8,
                                  (lshash >> 32) & ones_8,
                                  (lshash >> 40) & ones_8,
                                  (lshash >> 48) & ones_8,
                                  (lshash >> 56) & ones_8,
                                  (lshash >> 64) & ones_8,
                                  (lshash >> 72) & ones_8,
                                  (lshash >> 80) & ones_8,
                                  (lshash >> 88) & ones_8,
                                  (lshash >> 96) & ones_8,
                                  (lshash >> 104) & ones_8,
                                  (lshash >> 112) & ones_8,
                                  (lshash >> 120) & ones_8])
            else:
                assert (False)

        for index, row in enumerate(row_list):
            path_file = os.path.join(
                input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
            add_tuple_one_file(path_file, index)

    elif f == 256:
        def add_tuple_one_file(path_file, index):
            all_of_it = read_file(path_file)
            features = get_tokens(
                all_of_it, shingle_num, len_limit)
            lshash = Simhash(
                features, hashfunc=mySHA256, f=f).value
            if r == 4:
                LSH_tuple.append([index,
                                  lshash & ones_64,
                                  (lshash >> 64) & ones_64,
                                  (lshash >> 128) & ones_64,
                                  (lshash >> 192) & ones_64])
            elif r == 8:
                LSH_tuple.append([index,
                                  lshash & ones_32,
                                  (lshash >> 32) & ones_32,
                                  (lshash >> 64) & ones_32,
                                  (lshash >> 96) & ones_32,
                                  (lshash >> 128) & ones_32,
                                  (lshash >> 160) & ones_32,
                                  (lshash >> 192) & ones_32,
                                  (lshash >> 224) & ones_32])
            elif r == 16:
                LSH_tuple.append([index,
                                  lshash & ones_16,
                                  (lshash >> 16) & ones_16,
                                  (lshash >> 32) & ones_16,
                                  (lshash >> 48) & ones_16,
                                  (lshash >> 64) & ones_16,
                                  (lshash >> 80) & ones_16,
                                  (lshash >> 96) & ones_16,
                                  (lshash >> 112) & ones_16,
                                  (lshash >> 128) & ones_16,
                                  (lshash >> 144) & ones_16,
                                  (lshash >> 160) & ones_16,
                                  (lshash >> 176) & ones_16,
                                  (lshash >> 192) & ones_16,
                                  (lshash >> 208) & ones_16,
                                  (lshash >> 224) & ones_16])

            else:
                assert (False)

        for index, row in enumerate(row_list):
            path_file = os.path.join(
                input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
            add_tuple_one_file(path_file, index)
    else:
        assert (0)

    # use union-find data structure to
    # find the connected components of the graph
    uf = UnionFind(range(num_blobs))

    for i in range(1, len(LSH_tuple[0])):
        LSH_tuple.sort(key=lambda x: x[i])
        for j in range(len(LSH_tuple) - 1):
            if LSH_tuple[j][i] == LSH_tuple[j+1][i]:
                uf.union(LSH_tuple[j][0], LSH_tuple[j+1][0])

    sorted_row_list = []
    for connected_component in uf.components():
        list_connected_component = list(connected_component)
        row_list_connected_component = [row_list[i]
                                        for i in list_connected_component]
        sorted_row_list.extend(sorted(row_list_connected_component,
                                      key=lambda x: int(df.iloc[x]['length']), reverse=True))

    # assert (check_is_permutation_list(row_list, sorted_row_list))
    return row_list


def minhash_graph_unionfind_list(df, row_list, shingle_num, f, r, len_limit, input_dir):
    num_blobs = len(row_list)

    LSH_tuple = []

    b = f // r

    total_uncompressed_size = df.iloc[row_list]['length'].sum()
    # the list could contain few (big) files so
    if num_blobs < 3 or total_uncompressed_size < 32 * (2**20):  # 32MiB
        return sorted(row_list, key=lambda x: int(df.iloc[x]['length']), reverse=True)

    LSH_tuple = []

    def add_tuple_one_file(path_file, index):
        # all_of_it = read_file(path_file)
        # features = get_tokens(all_of_it, shingle_num, len_limit)
        m1 = read_file_get_minhash_on_tokens(
            path_file, shingle_num, len_limit, f)
        # m1 = MinHash(num_perm=f)
        # for d in features:
        #     m1.update(d.encode('utf8'))

        curr_tuple = [index]
        idx = 0
        for _ in range(r):
            curr_band = []
            for _ in range(b):
                curr_band.append(m1.hashvalues[idx])
                idx += 1

            curr_tuple.append(curr_band)

        LSH_tuple.append(curr_tuple)

    for index, row in enumerate(row_list):
        path_file = os.path.join(
            input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        add_tuple_one_file(path_file, index)

    # use union-find data structure to
    # find the connected components of the graph
    uf = UnionFind(range(num_blobs))

    for i in range(1, len(LSH_tuple[0])):
        LSH_tuple.sort(key=lambda x: x[i])
        for j in range(len(LSH_tuple) - 1):
            if LSH_tuple[j][i] == LSH_tuple[j+1][i]:
                uf.union(LSH_tuple[j][0], LSH_tuple[j+1][0])

    sorted_row_list = []
    for connected_component in uf.components():
        list_connected_component = list(connected_component)
        row_list_connected_component = [row_list[i]
                                        for i in list_connected_component]
        if byte_size_list_rows(df, row_list_connected_component) > 32 * (2**20) and len(row_list_connected_component) > 3:
            sorted_row_list.extend(tlsh_sort_list(
                df, row_list_connected_component, input_dir=input_dir))
        else:
            sorted_row_list.extend(sorted(row_list_connected_component,
                                          key=lambda x: int(df.iloc[x]['length']), reverse=True))

    # assert (check_is_permutation_list(row_list, sorted_row_list))
    return sorted_row_list

# Example of simhash
# 77237983905550514267427226421535267128310482787725109977399130815457050873635
# 24593597262581824657681274770535835476157267412892898867107059387508568986251
# 90341045248643906465381025721606941739423780834257608925848513060238636450902
# 61851553286930728096157589659269299368734783742729398493540466611450627577203
def simhash_sort(df, shingle_num, f, len_limit, input_dir):
    num_blobs = len(df.index)

    LSH = []

    # start_simhash_computing = time.time()
    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def compute_one_simhash(index, path_file, len_file):
            lshash = 0
            # check if size pf the file is < 1MiB
            if len_file < 2**20:
                # read all lines at once
                all_of_it = read_file(path_file)
                features = get_tokens(
                    all_of_it, shingle_num, len_limit)
                if f == 64:
                    lshash = Simhash(
                        features, hashfunc=spookyhash.hash64, f=f).value
                elif f == 128:
                    lshash = Simhash(
                        features, hashfunc=spookyhash.hash128, f=f).value
                elif f == 256:
                    lshash = Simhash(
                        features, hashfunc=mySHA256, f=f).value
                else:
                    assert (0)
                # lock.acquire()
                LSH.append([index, lshash])
                # lock.release()
            else:
                # lock.acquire()
                LSH.append([index, 0])
                # lock.release()

        for row in range(num_blobs):
            path_file = os.path.join(
                input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
            # print(path_file)
            len_file = int(df.iloc[row]['length'])
            executor.submit(compute_one_simhash, row, path_file, len_file)

    LSH.sort(key=lambda x: x[1])


    LSH_0 = [item[0] for item in LSH]

    # assert(len(LSH_0) == len(set(LSH_0)))
    # assert (check_is_permutation(LSH_0, num_blobs))
    return LSH_0


def simhash_sort_content(df, shingle_num, _f, input_dir):
    num_blobs = len(df.index)

    # lock = threading.Lock()
    LSH = []

    # start_simhash_computing = time.time()

    with ThreadPoolExecutor(NUM_THREAD) as executor:
        for index in range(num_blobs):
            def compute_one_simhash(index, path_file):
                lshash = 0
                # check if size pf the file is < 1MiB
                if int(df.iloc[index]['length']) < 2**20:
                    if _f == 64:
                        all_of_it = read_file(path_file)
                        features = get_fixed_num_features_content(
                            all_of_it)
                        lshash = Simhash(
                            features, hashfunc=spookyhash.hash64, f=_f).value
                    elif _f == 128:
                        all_of_it = read_file(path_file)
                        features = get_fixed_num_features_content(
                            all_of_it)
                        lshash = Simhash(
                            features, hashfunc=spookyhash.hash128, f=_f).value
                    elif _f == 256:
                        all_of_it = read_file(path_file)
                        features = get_fixed_num_features_content(
                            all_of_it)
                        lshash = Simhash(
                            features, hashfunc=mySHA256, f=_f).value
                    else:
                        assert (0)
                    # LSH[index] = [index, np.uint64(lshash)]
                    # lshash seems to be 128 bit
                    # lock.acquire()
                    LSH.append([index, lshash])
                    # lock.release()
                else:
                    # lock.acquire()
                    LSH.append([index, 0])
                    # lock.release()

            path_file = os.path.join(
                input_dir, df.iloc[index]['file_id'])
            executor.submit(compute_one_simhash, index, path_file)

    # time_simhash_computing = time.time() - start_simhash_computing

    # start_simhash_sorting = time.time()
    # LSH.sort(key=lambda x: x[1])
    LSH.sort(key=lambda x: x[1])

    # time_simhash_sorting = time.time() - start_simhash_sorting

    # print("simhash computing : " + str(time_simhash_computing))
    # print("simhash sorting : " + str(time_simhash_sorting))

    LSH_0 = [item[0] for item in LSH]

    assert (len(LSH_0) == len(set(LSH_0)))
    # assert (check_is_permutation(LSH_0, num_blobs))
    return LSH_0


def length_simhash_sort(df, shingle_num, _f, len_limit, div, input_dir):
    # df is the dataframe containing the blobs
    # shingle_num is the number of lines in a shingle
    # _f is the number of bits of the simhash
    # lines shorted than len_limit are not considered in the simhash
    # div is the constant we divide the len of the file to create groups of similar blobs

    num_blobs = len(df.index)
    # lock = threading.Lock()
    LSH = []
    # start_simhash_computing = time.time()
    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def compute_one_simhash(index, path_file):
            lshash = 0
            file_size_KiB = 0
            if int(df.iloc[index]['length']) < 2**20 and os.path.exists(path_file):
                all_of_it = read_file(path_file)
                file_size_KiB = len(all_of_it) // div
                features = get_tokens(all_of_it, shingle_num, len_limit)
                if _f == 64:
                    h_fun = spookyhash.hash64
                elif _f == 128:
                    h_fun = spookyhash.hash128
                elif _f == 256:
                    h_fun = mySHA256
                else:
                    assert (0)
                lshash = Simhash(
                    features, hashfunc=h_fun, f=_f).value
                # lock.acquire()
                # Lock not needed append in list is thread-safe
                LSH.append([index, file_size_KiB, lshash])
                # lock.release()

        for row in range(num_blobs):
            path_file = os.path.join(
                input_dir, df.iloc[row]['file_id'])
            executor.submit(compute_one_simhash, row, path_file)
            # compute_one_simhash(index, path_file)

    LSH.sort(key=lambda x: (x[1], gray_code(x[2])))
    LSH_0 = [item[0] for item in LSH]

    # assert (check_is_permutation(LSH_0, num_blobs))
    return LSH_0


def simhash_sort_graycode(df, shingle_num, _f, len_limit, input_dir):
    num_blobs = len(df.index)
    LSH = []

    # start_simhash_computing = time.time()
    for index in range(num_blobs):
        path_file = os.path.join(input_dir, df.iloc[index]['file_id'])
        lshash = 0
        if os.path.getsize(path_file) < 2**20:
            if _f == 64:
                all_of_it = read_file(path_file)
                features = get_tokens(all_of_it, shingle_num, len_limit)
                lshash = Simhash(
                    features, hashfunc=spookyhash.hash64, f=_f).value
            elif _f == 128:
                all_of_it = read_file(path_file)
                features = get_tokens(all_of_it, shingle_num, len_limit)
                lshash = Simhash(
                    features, hashfunc=spookyhash.hash128, f=_f).value
            elif _f == 256:
                all_of_it = read_file(path_file)
                features = get_tokens(all_of_it, shingle_num, len_limit)
                lshash = Simhash(features, hashfunc=mySHA256, f=_f).value
            else:
                assert (0)
            LSH.append([index, lshash])
        else:
            LSH.append([index, 0])

    # LSH.sort(key=lambda x: x[1])
    LSH.sort(key=lambda x: gray_code(x[1]))

    LSH_0 = [item[0] for item in LSH]

    # assert (check_is_permutation(LSH_0, num_blobs))
    return LSH_0

# Example of TLSH hash
# HEADER --> [T11A11B1] BODY --> [09394415807BF8216AED551D93199268411B2EECB3961D94681F648E3C5BD28B]
# T11701F281E44F5A633028487064AF5CF3608EE54643ED0E3C1556319D6683BE0424FFEE
# T12DC01269517FC97B9D040DF776484401D350A0C7502A6D8018A26D20190C65D58C7A66
# T10BE19427A7C4133E48A30251761E75ECA32A94B873B282107C6D5238B346D359BBFBDD
def TLSH_sort(df, input_dir):
    num_blobs = len(df.index)
    # LSH = [[0, 0] for _ in range(num_blobs)]
    LSH = []
    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def compute_one_tlsh(index, path_file):
            # read all lines at once
            all_of_it = read_file_bytes(path_file)

            # https://documents.trendmicro.com/assets/wp/wp-locality-sensitive-hash.pdf
            # Byte[0] = 'T'
            # Byte[1] = '1'
            # Byte[2]Byte[3] = chechsum
            # Byte[4]Byte[5] = log(lenght)
            # Byte[6]Byte[7] = constructed out of two 16 bit quantities derived from the quartiles: q1, q2 and q3
            lshash = tlsh.hash(all_of_it)[8:]
            #print(lshash)
            LSH.append([index, lshash])

        for row in range(num_blobs):
            path_file = os.path.join(
                input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
            # We skip the really few big files (<4MiB), that are going to be compressed togheter
            if int(df.iloc[row]['length']) < 2**22:
                executor.submit(compute_one_tlsh, row, path_file)
            else:
                LSH.append([row, '0'])

    LSH.sort(key=lambda x: x[1])
    LSH_0 = [item[0] for item in LSH]

    # assert (check_is_permutation(LSH_0, num_blobs))
    return LSH_0


def icf_tlsh_single_sort(df, input_dir):
    num_blobs = len(df.index)
    # LSH = [[0, 0] for _ in range(num_blobs)]
    LSH = []

    pattern = re.compile(rb'import\s+(\S.+?)|class\s+(\S+?):|def\s+(\S+?)\(.*?\):')

    def compute_one_tlsh(index, path_file):
        # read all lines at once
        with open(path_file, mode='rb') as file:
            chunk = file.read(2 ** 16)
        # chunk = read_file_size(path_file, 2 ** 16)

        # Get only the names
        names = b' '.join(b''.join(match) for match in pattern.findall(chunk))
        # print(all_of_it)

        # https://documents.trendmicro.com/assets/wp/wp-locality-sensitive-hash.pdf
        # Byte[0] = 'T'
        # Byte[1] = '1'
        # Byte[2]Byte[3] = chechsum
        # Byte[4]Byte[5] = log(lenght)
        # Byte[6]Byte[7] = constructed out of two 16 bit quantities derived from the quartiles: q1, q2 and q3
        lshash = tlsh.hash(names)[8:]
        # print(lshash)
        LSH.append([index, lshash])

    for row in range(num_blobs):
        path_file = os.path.join(input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        # We skip the really few big files (<4MiB), that are going to be compressed togheter
        if int(df.iloc[row]['length']) < 2 ** 22:
            compute_one_tlsh(row, path_file)
        else:
            LSH.append([row, '0'])

    LSH.sort(key=lambda x: x[1])
    LSH_0 = [item[0] for item in LSH]

    # assert (check_is_permutation(LSH_0, num_blobs))
    return LSH_0


def icf_single_sort(df, input_dir):
    num_blobs = len(df.index)
    # LSH = [[0, 0] for _ in range(num_blobs)]
    mapping = []

    pattern = re.compile(rb'import\s+(\S.+?)|class\s+(\S+?):|def\s+(\S+?)\(.*?\):')

    def compute_one_tlsh(index, path_file):
        # read all lines at once
        with open(path_file, mode='rb') as file:
            chunk = file.read(2 ** 16)

        # Get only the names
        names = b' '.join(b''.join(match) for match in pattern.findall(chunk))

        mapping.append([index, names])

    for row in range(num_blobs):
        path_file = os.path.join(input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        compute_one_tlsh(row, path_file)

    mapping.sort(key=lambda x: x[1])
    mapping_0 = [item[0] for item in mapping]

    # assert (check_is_permutation(mapping_0, num_blobs))
    return mapping_0


def oicf_single_sort(df, input_dir):
    num_blobs = len(df.index)
    # LSH = [[0, 0] for _ in range(num_blobs)]
    mapping = []

    pattern = re.compile(rb'import\s+(\S.+?)|class\s+(\S+?):|def\s+(\S+?)\(.*?\):')

    def compute_one_tlsh(index, path_file):
        # read all lines at once
        with open(path_file, mode='rb') as file:
            chunk = file.read(2 ** 16)

        # Get only the names
        names = tuple(sorted(b''.join(match) for match in pattern.findall(chunk)))

        mapping.append([index, names])

    for row in range(num_blobs):
        path_file = os.path.join(input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        compute_one_tlsh(row, path_file)

    mapping.sort(key=lambda x: x[1])
    mapping_0 = [item[0] for item in mapping]

    # assert (check_is_permutation(mapping_0, num_blobs))
    return mapping_0


def icf_multi_sort(df, input_dir):
    num_blobs = len(df.index)
    # LSH = [[0, 0] for _ in range(num_blobs)]
    mapping = []

    pattern = re.compile(rb'import\s+(\S.+?)|class\s+(\S+?):|def\s+(\S+?)\(.*?\):')

    def compute_one_tlsh(index, path_file):
        # read all lines at once
        with open(path_file, mode='rb') as file:
            chunk = file.read(2 ** 16)

        # Get only the names
        names = b' '.join(b''.join(match) for match in pattern.findall(chunk))

        mapping.append([index, names])

    with ThreadPoolExecutor(NUM_THREAD) as executor:
        for row in range(num_blobs):
            path_file = os.path.join(input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
            executor.submit(compute_one_tlsh, row, path_file)

    mapping.sort(key=lambda x: x[1])
    mapping_0 = [item[0] for item in mapping]

    # assert (check_is_permutation(mapping_0, num_blobs))
    return mapping_0


def cf_single_sort(df, input_dir):
    num_blobs = len(df.index)
    # LSH = [[0, 0] for _ in range(num_blobs)]
    mapping = []

    pattern = re.compile(rb'class\s+(\S+?):|def\s+(\S+?)\(.*?\):')

    def compute_one_tlsh(index, path_file):
        with open(path_file, mode='rb') as file:
            chunk = file.read(2 ** 16)

        # Get only the names
        names = b' '.join(b''.join(match) for match in pattern.findall(chunk))

        mapping.append([index, names])

    for row in range(num_blobs):
        path_file = os.path.join(input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        compute_one_tlsh(row, path_file)

    mapping.sort(key=lambda x: x[1])
    mapping_0 = [item[0] for item in mapping]

    # assert (check_is_permutation(mapping_0, num_blobs))
    return mapping_0


def ocf_single_sort(df, input_dir):
    num_blobs = len(df.index)
    # LSH = [[0, 0] for _ in range(num_blobs)]
    mapping = []

    pattern = re.compile(rb'class\s+(\S+?):|def\s+(\S+?)\(.*?\):')

    def compute_one_tlsh(index, path_file):
        with open(path_file, mode='rb') as file:
            chunk = file.read(2 ** 16)

        # Get only the names
        names = tuple(sorted(b''.join(match) for match in pattern.findall(chunk)))

        mapping.append([index, names])

    for row in range(num_blobs):
        path_file = os.path.join(input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        compute_one_tlsh(row, path_file)

    mapping.sort(key=lambda x: x[1])
    mapping_0 = [item[0] for item in mapping]

    # assert (check_is_permutation(mapping_0, num_blobs))
    return mapping_0

if sys.version_info < (3, 9):
    # Examples ssdeep hash
    # 24:X+cL58FKrmjNj5byfOkHhylUHTbVeJORhPjZRUkPH2hQ29utFgmOfdr8OkCZ:uEU2gN1byJHcMlj4AWhb2gvfdrtkS
    # 768:Vi2jrO1M4olm6bUAEkNzzTX4mdtsmYLn+G4TxwnbAnj09HiWZxpxxvWPPLZW5sMs:Y2j/o2LvImPfYb6qbqjAzvQGv+8f4182
    # 192:yodlQ3Y80SawnxbKe40qo3yuogby5jRsykoky0o9yEpNKLavj/YMR5B1ZmNUJlDw:lbQ3YZSaQ40qS1OjgXWnYMIo107G8
    # 3072:Aty7AhaX5VJNaoj6fsNcpvutfz+FrGe957faKVEWEtp3vceZBzhaYeZtFxptrvrT:Cy7AhaX5VJNRcNutfz+FrGe957faKVEM
    # 384:2LwSY5GAB1UnlLarSWzpPrIHVR3DFVm36/0XMRtIFIr//Rw3zHiNRI2BovL8+J10:2OGAB1slLaryfRI2BovL8+J1Kyl6m1MT
    # 24:YQ1LumbOkn/iKie5VRdRKvMa+uE/JqPs5C:duZoVIMa/E/gPs5C
    # 1536:pps4BG/wfCXpa3Kixq0TOht7meU1zSwFUrX:pvyZbU1zSwOr
    def ssdeep_sort(df, input_dir):
        num_blobs = len(df.index)
        LSH = []
        with ThreadPoolExecutor(NUM_THREAD) as executor:
            def compute_one_ssdeep(index, path_file):
                all_of_it = read_file_bytes(path_file)
                # lshash = ssdeep.hash(all_of_it)#[2:]
                lshash = ssdeep.hash(all_of_it)
                #print(lshash)
                #lshash = lshash.split(':')  # [1:2]
                # delete the first part of the hash, which is the blocksize
                #lshash = lshash[1] + ':' + lshash[2]
                #print(lshash)
                LSH.append([index, lshash])

            for row in range(num_blobs):
                path_file = os.path.join(
                    input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
                # We skip the really few big files (<4MiB), that are going to be compressed togheter
                if int(df.iloc[row]['length']) < 2**23:
                    executor.submit(compute_one_ssdeep, row, path_file)
                else:
                    LSH.append([row, '0'])

        LSH.sort(key=lambda x: x[1])
        LSH_0 = [item[0] for item in LSH]

        # assert (check_is_permutation(LSH_0, num_blobs))
        return LSH_0


def simhash_graph_technique_unionfind(df, shingle_num, f, r, len_limit, input_dir):
    num_blobs = len(df.index)

    LSH_tuple = []

    if f == 128:
        with ThreadPoolExecutor(NUM_THREAD) as executor:
            def add_tuple_one_file(path_file, index):
                all_of_it = read_file(path_file)
                features = get_tokens(all_of_it, shingle_num, len_limit)
                lshash = Simhash(
                    features, hashfunc=spookyhash.hash128, f=f).value
                if r == 4:
                    LSH_tuple.append([index,
                                      lshash & ones_32,
                                      (lshash >> 32) & ones_32,
                                      (lshash >> 64) & ones_32,
                                      (lshash >> 96) & ones_32])
                elif r == 8:
                    LSH_tuple.append([index,
                                      lshash & ones_16,
                                      (lshash >> 16) & ones_16,
                                      (lshash >> 32) & ones_16,
                                      (lshash >> 48) & ones_16,
                                      (lshash >> 64) & ones_16,
                                      (lshash >> 80) & ones_16,
                                      (lshash >> 96) & ones_16,
                                      (lshash >> 112) & ones_16])
                elif r == 16:
                    LSH_tuple.append([index,
                                      lshash & ones_8,
                                      (lshash >> 8) & ones_8,
                                      (lshash >> 16) & ones_8,
                                      (lshash >> 24) & ones_8,
                                      (lshash >> 32) & ones_8,
                                      (lshash >> 40) & ones_8,
                                      (lshash >> 48) & ones_8,
                                      (lshash >> 56) & ones_8,
                                      (lshash >> 64) & ones_8,
                                      (lshash >> 72) & ones_8,
                                      (lshash >> 80) & ones_8,
                                      (lshash >> 88) & ones_8,
                                      (lshash >> 96) & ones_8,
                                      (lshash >> 104) & ones_8,
                                      (lshash >> 112) & ones_8,
                                      (lshash >> 120) & ones_8])
                else:
                    assert (False)

            for index in range(num_blobs):
                path_file = os.path.join(
                    input_dir, df.iloc[index]['file_id'])
                if int(df.iloc[index]['length']) < 2**20:
                    executor.submit(add_tuple_one_file, path_file, index)
                    # add_tuple_one_file(path_file)
                else:
                    LSH_tuple_tmp = [index]
                    for i in range(r):
                        LSH_tuple_tmp.append(0)
                    LSH_tuple.append(LSH_tuple_tmp)
    elif f == 256:
        with ThreadPoolExecutor(NUM_THREAD) as executor:
            def add_tuple_one_file(path_file, index):
                all_of_it = read_file(path_file)
                features = get_tokens(all_of_it, shingle_num, len_limit)
                lshash = Simhash(features, hashfunc=mySHA256, f=f).value
                if r == 4:
                    LSH_tuple.append([index,
                                      lshash & ones_64,
                                      (lshash >> 64) & ones_64,
                                      (lshash >> 128) & ones_64,
                                      (lshash >> 192) & ones_64])
                elif r == 8:
                    LSH_tuple.append([index,
                                      lshash & ones_32,
                                      (lshash >> 32) & ones_32,
                                      (lshash >> 64) & ones_32,
                                      (lshash >> 96) & ones_32,
                                      (lshash >> 128) & ones_32,
                                      (lshash >> 160) & ones_32,
                                      (lshash >> 192) & ones_32,
                                      (lshash >> 224) & ones_32])
                elif r == 16:
                    LSH_tuple.append([index,
                                      lshash & ones_16,
                                      (lshash >> 16) & ones_16,
                                      (lshash >> 32) & ones_16,
                                      (lshash >> 48) & ones_16,
                                      (lshash >> 64) & ones_16,
                                      (lshash >> 80) & ones_16,
                                      (lshash >> 96) & ones_16,
                                      (lshash >> 112) & ones_16,
                                      (lshash >> 128) & ones_16,
                                      (lshash >> 144) & ones_16,
                                      (lshash >> 160) & ones_16,
                                      (lshash >> 176) & ones_16,
                                      (lshash >> 192) & ones_16,
                                      (lshash >> 208) & ones_16,
                                      (lshash >> 224) & ones_16])

                else:
                    assert (False)
            for index in range(num_blobs):
                path_file = os.path.join(
                    input_dir, df.iloc[index]['local_path'], df.iloc[index]['file_id'])
                # and os.path.exists(path_file):
                if int(df.iloc[index]['length']) < 2**20:
                    executor.submit(add_tuple_one_file, path_file, index)
                    # add_tuple_one_file(path_file)
                else:
                    LSH_tuple_tmp = [index]
                    for i in range(r):
                        LSH_tuple_tmp.append(0)
                    LSH_tuple.append(LSH_tuple_tmp)
    else:
        assert (0)

    # use union-find data structure to
    # find the connected components of the graph
    uf = UnionFind(range(num_blobs))

    for i in range(1, len(LSH_tuple[0])):
        LSH_tuple.sort(key=lambda x: x[i])
        for j in range(len(LSH_tuple) - 1):
            if LSH_tuple[j][i] == LSH_tuple[j+1][i]:
                uf.union(LSH_tuple[j][0], LSH_tuple[j+1][0])

    row_list = []
    for connected_component in uf.components():
        list_connected_component = list(connected_component)
        if byte_size_list_rows(df, list_connected_component) > 32 * (2**20) and len(list_connected_component) > 3:
            # clutered_row_list = minhash_cluster_list(
            #   df, connected_component, shingle_num, _f, len_limit, 2**20, {'max_iter': 3500})
            # sorted_row_list = simhash_sort_list(
            #    df, connected_component, shingle_num, _f, len_limit)
            sorted_row_list = tlsh_sort_list(
                df, connected_component, input_dir=input_dir)
            row_list.extend(sorted_row_list)
        else:
            row_list.extend(sorted(list_connected_component, key=lambda x: int(
                df.iloc[x]['length']), reverse=True))

    # assert (check_is_permutation(row_list, num_blobs))
    return row_list


def minhash_graph_technique_unionfind(df, shingle_num, f, r, len_limit, input_dir):
    num_blobs = len(df.index)

    LSH_tuple = []

    # TODO: what if r doesn't divide f
    # just dont consider the rest
    b = f // r

    def add_tuple_one_file(path_file, index):
        # all_of_it = read_file(path_file)
        # features = get_tokens(all_of_it, shingle_num, len_limit)
        # m1 = MinHash(num_perm=f)
        # for d in features:
        #    m1.update(d.encode('utf8'))

        m1 = read_file_get_minhash_on_tokens(
            path_file, shingle_num, len_limit, f)

        curr_tuple = [index]
        idx = 0
        for _ in range(r):
            curr_band = []
            for _ in range(b):
                curr_band.append(m1.hashvalues[idx])
                idx += 1

            curr_tuple.append(curr_band)

        LSH_tuple.append(curr_tuple)

    start_time = time.time()

    # with ProcessPoolExecutor(NUM_THREAD) as executor:
    with ThreadPoolExecutor(NUM_THREAD) as executor:
        for index in range(num_blobs):
            path_file = os.path.join(
                input_dir, df.iloc[index]['local_path'], df.iloc[index]['file_id'])
            # and os.path.exists(path_file):
            file_size = int(df.iloc[index]['length'])
            if file_size > 2**20:
                simbol = 0
            # if 200 <= file_size <= 2**20:
            if file_size <= 2**20:
                executor.submit(add_tuple_one_file, path_file, index)
                # add_tuple_one_file(path_file)
            else:
                curr_tuple = [index]
                for _ in range(r):
                    curr_band = []
                    for _ in range(b):
                        curr_band.append([simbol])
                    curr_tuple.append(curr_band)

                LSH_tuple.append(curr_tuple)

    start_time = time.time()
    uf = UnionFind(range(num_blobs))

    # each list of Minhash is divided into r groups of b integer each

    # for each group
    for i in range(1, len(LSH_tuple[0])):
        # sort by the group
        LSH_tuple.sort(key=lambda x: x[i])
        # so that we can groupd togheter the ones that are equal
        for j in range(len(LSH_tuple) - 1):
            if LSH_tuple[j][i] == LSH_tuple[j+1][i]:
                uf.union(LSH_tuple[j][0], LSH_tuple[j+1][0])

    row_list = []
    num_connected_components = len(uf.components())

    # do this in parallel? Nope, it uses 1/100 of the time needed by add_tuple_one_file
    for connected_component in uf.components():
        list_connected_component = list(connected_component)
        if byte_size_list_rows(df, list_connected_component) > 32 * (2**20) and len(list_connected_component) > 5:
            sorted_row_list = tlsh_sort_list(
                df, connected_component, input_dir=input_dir)
            row_list.extend(sorted_row_list)
        else:
            row_list.extend(sorted(list_connected_component, key=lambda x: int(
                df.iloc[x]['length']), reverse=True))

    # print(f'building union-find time(s) {time.time() - start_time}')
    # print(f'num_connected_components {num_connected_components} num_blobs {num_blobs}')
    # assert (check_is_permutation(row_list, num_blobs))
    return row_list

def simhash_from_file(file_path, shingle_num, _f):
    all_of_it = read_file(file_path)
    features = get_tokens(all_of_it, shingle_num)
    return Simhash(features, hashfunc=spookyhash.hash128, f=_f).value

# the function uses guessing_fun1 as file type then it permutes the blobs in the same cluster using the row_sorting_fun
def hybrid_type_1guess(df, guessing_fun1, row_sorting_fun, input_dir):
    lock = threading.Lock()

    map_type_rows = {}
    num_blobs = len(df.index)

    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def compute_one_file_type(file_path, file_size, index):
            if file_size > 2**20:
                file_type = 'too_big'
            elif file_size < 200:
                file_type = 'too_small'
            else:
                file_type = guessing_fun1(file_path)

            # print(file_type)
            lock.acquire()
            if file_type not in map_type_rows:
                map_type_rows[file_type] = [index]
            else:
                map_type_rows[file_type].append(index)
            lock.release()

        for index in range(num_blobs):
            file_path = os.path.join(input_dir, df.iloc[index]['local_path'], df.iloc[index]['file_id'])
            file_size = int(df.iloc[index]['length'])
            executor.submit(compute_one_file_type, file_path, file_size, index)
            # compute_one_file_type(file_path, file_size, index)
        # print('Computed mappping file->type(mime/guesslang)')

    permutation = []

    for t, row_list in map_type_rows.items():
        if row_sorting_fun != None and ((byte_size_list_rows(df, row_list) > (2 * (2**20))) and len(row_list) > 3):
            tmp = row_sorting_fun(
                df=df, row_list=row_list, input_dir=input_dir)
            # assert (check_is_permutation_list(tmp, row_list))
            permutation.extend(tmp)
        else:
            tmp = sorted(row_list, key=lambda x: int(
                df.iloc[x]['length']), reverse=True)
            # assert (check_is_permutation_list(tmp, row_list))
            permutation.extend(tmp)

    # print('Merged cluster and sorted by size')
    # assert (check_is_permutation(permutation, num_blobs))
    return permutation


# the function uses guessing_fun1 as file type, if the type contains string_pattern, is uses guessing_fun2
def hybrid_type_new(df, guessing_fun1, string_pattern, guessing_fun2, row_sorting_fun, input_dir):
    lock = threading.Lock()

    # TODO: refine mapping with https://guesslang.readthedocs.io/en/latest/
    map_type_rows = {}
    num_blobs = len(df.index)

    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def compute_one_file_type(file_path, file_size, index):
            if file_size > 2**20:
                file_type = 'too_big'
            elif file_size < 200:
                file_type = 'too_small'
            else:
                file_type = guessing_fun1(file_path)
                if string_pattern in file_type:
                    file_type = guessing_fun2(file_path)

            lock.acquire()
            if file_type not in map_type_rows:
                map_type_rows[file_type] = [index]
            else:
                map_type_rows[file_type].append(index)
            lock.release()

        for index in range(num_blobs):
            file_path = os.path.join(
                input_dir, df.iloc[index]['local_path'], df.iloc[index]['file_id'])
            file_size = int(df.iloc[index]['length'])
            executor.submit(compute_one_file_type, file_path, file_size, index)
            # compute_one_file_type(file_path, file_size, index)
        # print('Computed mappping file->type(mime/guesslang)')

    permutation = []

    for t, row_list in map_type_rows.items():
        if (byte_size_list_rows(df, row_list) > (2 * (2**20))) and len(row_list) > 3:
            tmp = row_sorting_fun(
                df=df, row_list=row_list, input_dir=input_dir)
            # assert (check_is_permutation_list(tmp, row_list))
            permutation.extend(tmp)
        else:
            tmp = sorted(row_list, key=lambda x: int(
                df.iloc[x]['length']), reverse=True)
            # assert (check_is_permutation_list(tmp, row_list))
            permutation.extend(tmp)

    # print('Merged cluster and sorted by size')
    # assert (check_is_permutation(permutation, num_blobs))
    return permutation


def stats_simhash(df, shingle_num, _f, len_limit, input_dir):
    num_blobs = len(df.index)

    LSH = []

    for index in range(num_blobs):
        path_file = os.path.join(input_dir, df.iloc[index]['file_id'])
        # read all lines at once
        all_of_it = read_file(path_file)
        features = get_tokens(all_of_it, shingle_num, len_limit)
        lshash = 0
        if _f == 64:
            lshash = Simhash(features, hashfunc=spookyhash.hash64, f=_f).sums
        elif _f == 128:
            lshash = Simhash(features, hashfunc=spookyhash.hash128, f=_f).sums
        elif _f == 256:
            lshash = Simhash(features, hashfunc=mySHA256, f=_f).sums
        else:
            assert (0)

        if isinstance(lshash, np.ndarray):
            LSH.append(lshash)

    value_freq = {}
    for lsh in LSH:
        for i in lsh:
            if i not in value_freq:
                value_freq[i] = 1
            else:
                value_freq[i] += 1
    counter = [value_freq[x] for x in sorted(value_freq.keys())]

    plt.plot(range(len(counter)), counter)
    plt.yscale('log')
    plt.savefig("test.png")
    plt.close()


def get_stats_from_filename(df):
    lock = threading.Lock()

    map_type_rows = {}
    num_blobs = len(df.index)

    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def compute_one_file_extension(file_name):
            extension = file_name.split('.')[-1]

            lock.acquire()
            if extension not in map_type_rows:
                map_type_rows[extension] = 1
            else:
                map_type_rows[extension] += 1
            lock.release()

        for index in range(num_blobs):
            file_name = df.iloc[index]['filename']
            executor.submit(compute_one_file_extension, file_name)

    return map_type_rows


def get_stats_from_type(df, guessing_fun1, input_dir):
    lock = threading.Lock()

    map_type_rows = {}
    num_blobs = len(df.index)

    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def compute_one_file_type(file_path, file_size):
            if file_size > 2**20:
                file_type = 'too_big'
            elif file_size < 200:
                file_type = 'too_small'
            else:
                file_type = guessing_fun1(file_path)

            lock.acquire()
            if file_type not in map_type_rows:
                map_type_rows[file_type] = 1
            else:
                map_type_rows[file_type] += 1
            lock.release()

        for index in range(num_blobs):
            file_path = os.path.join(
                input_dir, df.iloc[index]['local_path'], df.iloc[index]['file_id'])
            file_size = int(df.iloc[index]['length'])
            executor.submit(compute_one_file_type, file_path, file_size)

    return map_type_rows


def print_stats(stats, method_name):
    print(f'Using {method_name} as file grouping method')
    print(f'we get {len(stats)} different groups')
    print(f'here they are with they respective size:')

    for k, v in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f'- group "{k}" --> {v} blobs')

    print()


# FROM https://github.com/deehzee/unionfind
"""
A union-find disjoint set data structure.

"""

# 2to3 sanity
# Third-party libraries
class UnionFind(object):
    """Union-find disjoint sets datastructure.

    Union-find is a data structure that maintains disjoint set
    (called connected components or components in short) membership,
    and makes it easier to merge (union) two components, and to find
    if two elements are connected (i.e., belong to the same
    component).

    This implements the "weighted-quick-union-with-path-compression"
    union-find algorithm.  Only works if elements are immutable
    objects.

    Worst case for union and find: :math:`(N + M \log^* N)`, with
    :math:`N` elements and :math:`M` unions. The function
    :math:`\log^*` is the number of times needed to take :math:`\log`
    of a number until reaching 1. In practice, the amortized cost of
    each operation is nearly linear [1]_.

    Terms
    -----
    Component
        Elements belonging to the same disjoint set

    Connected
        Two elements are connected if they belong to the same component.

    Union
        The operation where two components are merged into one.

    Root
        An internal representative of a disjoint set.

    Find
        The operation to find the root of a disjoint set.

    Parameters
    ----------
    elements : NoneType or container, optional, default: None
        The initial list of elements.

    Attributes
    ----------
    n_elts : int
        Number of elements.

    n_comps : int
        Number of distjoint sets or components.

    Implements
    ----------
    __len__
        Calling ``len(uf)`` (where ``uf`` is an instance of ``UnionFind``)
        returns the number of elements.

    __contains__
        For ``uf`` an instance of ``UnionFind`` and ``x`` an immutable object,
        ``x in uf`` returns ``True`` if ``x`` is an element in ``uf``.

    __getitem__
        For ``uf`` an instance of ``UnionFind`` and ``i`` an integer,
        ``res = uf[i]`` returns the element stored in the ``i``-th index.
        If ``i`` is not a valid index an ``IndexError`` is raised.

    __setitem__
        For ``uf`` and instance of ``UnionFind``, ``i`` an integer and ``x``
        an immutable object, ``uf[i] = x`` changes the element stored at the
        ``i``-th index. If ``i`` is not a valid index an ``IndexError`` is
        raised.

    .. [1] http://algs4.cs.princeton.edu/lectures/

    """

    def __init__(self, elements=None):
        self.n_elts = 0  # current num of elements
        self.n_comps = 0  # the number of disjoint sets or components
        self._next = 0  # next available id
        self._elts = []  # the elements
        self._indx = {}  # dict mapping elt -> index in _elts
        self._par = []  # parent: for the internal tree structure
        self._siz = []  # size of the component - correct only for roots

        if elements is None:
            elements = []
        for elt in elements:
            self.add(elt)

    def __repr__(self):
        return (
            '<UnionFind:\n\telts={},\n\tsiz={},\n\tpar={},\nn_elts={},n_comps={}>'
            .format(
                self._elts,
                self._siz,
                self._par,
                self.n_elts,
                self.n_comps,
            ))

    def __len__(self):
        return self.n_elts

    def __contains__(self, x):
        return x in self._indx

    def __getitem__(self, index):
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        return self._elts[index]

    def __setitem__(self, index, x):
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        self._elts[index] = x

    def add(self, x):
        """Add a single disjoint element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        None

        """
        if x in self:
            return
        self._elts.append(x)
        self._indx[x] = self._next
        self._par.append(self._next)
        self._siz.append(1)
        self._next += 1
        self.n_elts += 1
        self.n_comps += 1

    def find(self, x):
        """Find the root of the disjoint set containing the given element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        int
            The (index of the) root.

        Raises
        ------
        ValueError
            If the given element is not found.

        """
        if x not in self._indx:
            raise ValueError('{} is not an element'.format(x))

        p = self._indx[x]
        while p != self._par[p]:
            # path compression
            q = self._par[p]
            self._par[p] = self._par[q]
            p = q
        return p

    def connected(self, x, y):
        """Return whether the two given elements belong to the same component.

        Parameters
        ----------
        x : immutable object
        y : immutable object

        Returns
        -------
        bool
            True if x and y are connected, false otherwise.

        """
        return self.find(x) == self.find(y)

    def union(self, x, y):
        """Merge the components of the two given elements into one.

        Parameters
        ----------
        x : immutable object
        y : immutable object

        Returns
        -------
        None

        """
        # Initialize if they are not already in the collection
        for elt in [x, y]:
            if elt not in self:
                self.add(elt)

        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self._siz[xroot] < self._siz[yroot]:
            self._par[xroot] = yroot
            self._siz[yroot] += self._siz[xroot]
        else:
            self._par[yroot] = xroot
            self._siz[xroot] += self._siz[yroot]
        self.n_comps -= 1

    def component(self, x):
        """Find the connected component containing the given element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        set

        Raises
        ------
        ValueError
            If the given element is not found.

        """
        if x not in self:
            raise ValueError('{} is not an element'.format(x))
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        return set(elts[roots == self.find(x)])

    def components(self):
        """Return the list of connected components.

        Returns
        -------
        list
            A list of sets.

        """
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        distinct_roots = set(roots)
        return [set(elts[roots == root]) for root in distinct_roots]
        # comps = []
        # for root in distinct_roots:
        #     mask = (roots == root)
        #     comp = set(elts[mask])
        #     comps.append(comp)
        # return comps

    def component_mapping(self):
        """Return a dict mapping elements to their components.

        The returned dict has the following semantics:

            `elt -> component containing elt`

        If x, y belong to the same component, the comp(x) and comp(y)
        are the same objects (i.e., share the same reference). Changing
        comp(x) will reflect in comp(y).  This is done to reduce
        memory.

        But this behaviour should not be relied on.  There may be
        inconsitency arising from such assumptions or lack thereof.

        If you want to do any operation on these sets, use caution.
        For example, instead of

        ::

            s = uf.component_mapping()[item]
            s.add(stuff)
            # This will have side effect in other sets

        do

        ::

            s = set(uf.component_mapping()[item]) # or
            s = uf.component_mapping()[item].copy()
            s.add(stuff)

        or

        ::

            s = uf.component_mapping()[item]
            s = s | {stuff}  # Now s is different

        Returns
        -------
        dict
            A dict with the semantics: `elt -> component contianing elt`.

        """
        elts = np.array(self._elts)
        vfind = np.vectorize(self.find)
        roots = vfind(elts)
        distinct_roots = set(roots)
        comps = {}
        for root in distinct_roots:
            mask = (roots == root)
            comp = set(elts[mask])
            comps.update({x: comp for x in comp})
            # Change ^this^, if you want a different behaviour:
            # If you don't want to share the same set to different keys:
            # comps.update({x: set(comp) for x in comp})
        return comps
