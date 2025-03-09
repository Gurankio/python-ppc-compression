import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import spookyhash
import tlsh
from datasketch import MinHash
from matplotlib import pyplot as plt
from simhash import Simhash

from utils.unionfind import UnionFind

# check if python version is 3.8 or higher
if sys.version_info < (3, 9):
    import ssdeep

is_MinHash = False

start_block_size_exp = 21
end_block_size_exp = 25

start_clusters_div_exp = 26
end_clusters_div_exp = 27

float_1GiB = float(2 ** 30)
float_1MiB = float(2 ** 20)
float_1KiB = float(2 ** 10)

ones_8 = 2 ** 8 - 1
ones_16 = 2 ** 16 - 1
ones_32 = 2 ** 32 - 1
ones_64 = 2 ** 64 - 1

len_limit = 10


def write_filenames_to_file(list_to_write, filename):
    with open(filename, mode='w') as file:
        file.write('\n'.join(list_to_write))


def _diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def check_is_permutation(arr, n):
    # Set to check the count
    # of non-repeating elements
    s = set()
    maxEle = 0
    if len(arr) != n:
        print('len(arr) = {} != n = {}'.format(len(arr), n))
        print(_diff(arr, range(n)))
        return False

    for i in range(n):
        # Insert all elements in the set
        s.add(arr[i])
        # Calculating the max element
        maxEle = max(maxEle, arr[i])
    if maxEle != n - 1:
        print("max elem not equal n-1")
        print(_diff(arr, range(n)))
        return False

    # Check if set size is equal to n
    if len(s) == n:
        return True

    print("the array contains duplicates")
    print(_diff(arr, range(n)))
    return False


def check_is_permutation_list(arr1, arr2):
    if len(arr1) != len(arr2):
        print("len(arr1) = {} != len(arr2) = {}".format(len(arr1), len(arr2)))
        print(_diff(arr1, arr2))
        return False
    res = Counter(arr1) == Counter(arr2)
    if not res:
        print("the arrays are not permutations")
        print(_diff(arr1, arr2))
        return False
    else:
        return True


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


def get_tokens(file_content, width=1, len_limit=10):
    # get a list of lines from the content of the file
    tokens = file_content.split('\n')
    if width > 1:
        # tokes are consecutive lines grouped togheter
        tokens = [tokens[i:i + width][0]
                  for i in range(max(len(tokens) - width + 1, 1))]
    # remove lines with less than 10 chars and delete leading and trailing tabs and whitespaces
    tokens = [x.strip() for x in tokens if len(x) > len_limit]
    return tokens


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
            tokens = [tokens[i:i + width][0]
                      for i in range(max(len(tokens) - width + 1, 1))]
        # remove lines with less than 10 chars and delete leading and trailing tabs and whitespaces
        tokens = [x.strip() for x in tokens if len(x) > len_limit]
        m1 = MinHash(num_perm=f)
        for d in tokens:
            m1.update(d.encode('utf8'))

        return m1


def read_file_bytes(filename):
    with open(filename, mode='rb') as file:
        all_of_it = file.read()
        return all_of_it


def byte_size_list_rows(df, row_list):
    to_return = 0
    for row in row_list:
        to_return += df.iloc[row]['length']

    return to_return


def tlsh_sort_list(df, row_list, input_dir):
    LSH = []
    for index, row in enumerate(row_list):
        path_file = os.path.join(input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
        file_size = df.iloc[row]['length']
        lshash = '0'
        # TLSH is fast so limit at 4MiB
        if file_size < 2 ** 22:
            # read all lines at once
            all_of_it = read_file_bytes(path_file)
            lshash = tlsh.hash(all_of_it)[8:]
        LSH.append([row, lshash])

    LSH.sort(key=lambda x: x[1])
    LSH_0 = [item[0] for item in LSH]
    # assert (check_is_permutation_list(LSH_0, row_list))
    return LSH_0


def row_minhashgraph_unionfind_fun(df, row_list, input_dir):
    return minhash_graph_unionfind_list(df, row_list, 1, 256, 64, len_limit, input_dir)


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
            if LSH_tuple[j][i] == LSH_tuple[j + 1][i]:
                uf.union(LSH_tuple[j][0], LSH_tuple[j + 1][0])

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
    if num_blobs < 3 or total_uncompressed_size < 32 * (2 ** 20):  # 32MiB
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
            if LSH_tuple[j][i] == LSH_tuple[j + 1][i]:
                uf.union(LSH_tuple[j][0], LSH_tuple[j + 1][0])

    sorted_row_list = []
    for connected_component in uf.components():
        list_connected_component = list(connected_component)
        row_list_connected_component = [row_list[i]
                                        for i in list_connected_component]
        if byte_size_list_rows(df, row_list_connected_component) > 32 * (2 ** 20) and len(
                row_list_connected_component) > 3:
            sorted_row_list.extend(tlsh_sort_list(
                df, row_list_connected_component, input_dir=input_dir))
        else:
            sorted_row_list.extend(sorted(row_list_connected_component,
                                          key=lambda x: int(df.iloc[x]['length']), reverse=True))

    # assert (check_is_permutation_list(row_list, sorted_row_list))
    return sorted_row_list


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
            # print(lshash)
            LSH.append([index, lshash])

        for row in range(num_blobs):
            path_file = os.path.join(
                input_dir, df.iloc[row]['local_path'], df.iloc[row]['file_id'])
            # We skip the really few big files (<4MiB), that are going to be compressed togheter
            if int(df.iloc[row]['length']) < 2 ** 22:
                executor.submit(compute_one_tlsh, row, path_file)
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
                if int(df.iloc[index]['length']) < 2 ** 20:
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
                if int(df.iloc[index]['length']) < 2 ** 20:
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
            if LSH_tuple[j][i] == LSH_tuple[j + 1][i]:
                uf.union(LSH_tuple[j][0], LSH_tuple[j + 1][0])

    row_list = []
    for connected_component in uf.components():
        list_connected_component = list(connected_component)
        if byte_size_list_rows(df, list_connected_component) > 32 * (2 ** 20) and len(list_connected_component) > 3:
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
            if file_size > 2 ** 20:
                simbol = 0
            # if 200 <= file_size <= 2**20:
            if file_size <= 2 ** 20:
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
            if LSH_tuple[j][i] == LSH_tuple[j + 1][i]:
                uf.union(LSH_tuple[j][0], LSH_tuple[j + 1][0])

    row_list = []
    num_connected_components = len(uf.components())

    # do this in parallel? Nope, it uses 1/100 of the time needed by add_tuple_one_file
    for connected_component in uf.components():
        list_connected_component = list(connected_component)
        if byte_size_list_rows(df, list_connected_component) > 32 * (2 ** 20) and len(list_connected_component) > 5:
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
            if file_size > 2 ** 20:
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
        if row_sorting_fun != None and ((byte_size_list_rows(df, row_list) > (2 * (2 ** 20))) and len(row_list) > 3):
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
            if file_size > 2 ** 20:
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
