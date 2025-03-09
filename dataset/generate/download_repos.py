#!/usr/bin/env python3

import multiprocessing as mp
import os
import shutil
import subprocess as sp
import sys
import tempfile

import pandas as pd


def blob_size(blob):
    return len(sp.check_output(['git', 'cat-file', 'blob', blob[0]]))


def blob_save(blob, blobs_target_dir):
    out = sp.check_output(['git', 'cat-file', 'blob', blob[0]])
    with open(os.path.join(blobs_target_dir, blob[0]), 'wb') as w:
        w.write(out)
    return len(out)


# Retrieves all blobs from a repository before a given commit.
# If blobs_target_dir is not None, saves the blobs to that directory.
# Returns the size of all blobs and a list of (blob_hash, blob_path, blob_bytes) tuples.
def get_blobs(repo_url, commit, target_dir, blobs_target_dir=None):
    sp.run(['git', 'clone', '-q', repo_url, target_dir], check=True)
    cwd = os.getcwd()
    os.chdir(target_dir)
    args = ['git', 'rev-list', '--objects',
            '--filter=object:type=blob', commit]
    blobs = [(s[:40], s[41:]) for s in sp.check_output(
        args).decode('utf-8', 'ignore').split('\n')[1:-1]]
    pool = mp.Pool(mp.cpu_count())

    if blobs_target_dir is not None:
        d = os.path.join(cwd, blobs_target_dir) if not os.path.isabs(
            blobs_target_dir) else blobs_target_dir
        os.makedirs(d, exist_ok=True)
        results = pool.starmap(blob_save, ((o, d) for o in blobs))
    else:
        results = pool.map(blob_size, blobs)

    os.chdir(cwd)
    return sum(results), [(h, p, b) for (h, p), b in zip(blobs, results)]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: {} <repo_list1.csv> <repo_list2.csv> ...'.format(
            sys.argv[0]))
        sys.exit(1)

    OUTPUT_DIR = '/home/swh/50GiB_github/ALL'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)

    # df['language'] = df['language'].fillna('')

    # Lines below are for computing the size of all blobs in a repository a printing a csv on screen
    # print('repo_name,language,repo_url,last_commit,repo_size,blob_objects')
    # for row in df.itertuples():
    #     tmp_dir = tempfile.mkdtemp()
    #     repo_size, blobs = get_blobs(row.repo_url, row.last_commit, tmp_dir)
    #     print(f'{row.repo_name},{row.language},{row.repo_url},{row.last_commit},{repo_size},{len(blobs)}')
    #     shutil.rmtree(tmp_dir)

    # Lines below are for downloading all blobs in a repository
    used_dirs = dict()

    repos_dir = os.path.join(OUTPUT_DIR, 'repos')
    blobs_dir = os.path.join(OUTPUT_DIR, 'blobs')

    os.makedirs(repos_dir, exist_ok=True)
    os.makedirs(blobs_dir, exist_ok=True)
    for row in df.itertuples():
        dir_name = row.repo_name if row.repo_name not in used_dirs else f'{row.repo_name}_{used_dirs[row.repo_name]}'
        print(dir_name)
        used_dirs[row.repo_name] = used_dirs.get(row.repo_name, 0) + 1
        dst_repo_dir = os.path.join(repos_dir, dir_name)
        dst_blob_dir = os.path.join(blobs_dir, dir_name)
        repo_size, blobs = get_blobs(
            row.repo_url, row.last_commit, dst_repo_dir, dst_blob_dir)
        with open(os.path.join(blobs_dir, f'{dir_name}.csv'), 'w') as f:
            f.write('blob_hash,blob_bytes,blob_path\n')
            for blob_hash, blob_path, blob_bytes in blobs:
                f.write(f'{blob_hash},{blob_bytes},"{blob_path}"\n')
            total_size = sum(blob_bytes for _, _, blob_bytes in blobs)
        print('Downloaded {} blobs {} MiB from {} to {}'.format(
            len(blobs), round(repo_size / (2**20), 2), row.repo_url, dst_blob_dir))

    print(f"Done. Total size {total_size / (2**30)} GiB")

# print(get_blobs("https://github.com/sharkdp/bat", "0cc4e98560077a937c288e37988f676644e6e147", "/Users/giorgio/Downloads/bat", "/Users/giorgio/Downloads/batyblobs"))
