#!/usr/bin/env python3

import sys
import os
import subprocess
import argparse


Description = """
Download the archives created by the bench_PPC.py tool.
./download_dataset.py -s <size> -l <languages> -o <out-dir> -v

Example: 
./download_dataset.py -s DEBUG -o ./tmp
download into ./tmp 3 small debug datasets:
1) random_small contains 500 blobs (total uncompressed size 34MiB), 
2) Python_small contains 3.8K blobs (total uncompressed size 75MiB), 
3) C_small contains 7.6K blobs (total uncompressed size 259MiB)

./download_dataset.py -s 25GiB -l Python C -o ./tmp
download into ./tmp the 25GiB Python and C/C++ dataset

./download_dataset.py -s 50GiB -o ./tmp
download into ./tmp the 50GiB dataset (no language must be specified because this dataset is made from the most popular GitHub repositories)

./download_dataset.py -s 200GiB -l all -o ./tmp
download into ./tmp the ALL 200GiB datasets

The possible choices are for the size:
['DEBUG', '25GiB', '50GiB', '200GiB']

In case the size are 25GiB or 200GiB, the possible choices are for the language:
['C', 'Python', 'Javascript', 'Java', 'random', 'all']

./download_dataset.py -s 50GiB -o ./tmp
download into ./tmp the 50GiB dataset from most popular repositories from GitHub
"""

USE_GDRIVE = False

# if USE_GDRIVE == FALSE, the archives are downloaded from the following location
# No backslash in the end
ARCHIVES_LOCATION = 'https://pages.di.unipi.it/boffa/swh_endpoint'

def exec_cmd_bool(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    process.wait()
    output, _ = process.communicate()
    if process.returncode == 0:
        print('[OK]' + output.decode('utf-8'))
        return True
    else:
        print('[ERROR] (some archives migth not be available)' + output.decode('utf-8'))
        return False


# check comand line parameters and launch decompression
if __name__ == "__main__":

    # Instantiate the parser: options and arguments are in the global variable args
    parser = argparse.ArgumentParser(
        description=Description, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', '--size', type=str, default='DEBUG',
                        help='size of the archive you want to download',
                        choices=['DEBUG', '25GiB', '50GiB', '200GiB'])
    parser.add_argument('-l', '--languages',  nargs='+', default=['random'],
                        help='languages of the archive you want to download',
                        choices=['C', 'Python', 'Javascript', 'Java', 'random'])
    parser.add_argument('-o', '--output', type=str, default='.',
                        help='output directory')
    parser.add_argument('-v', action='store_true', default=False,
                        help='verbose output')

    args = parser.parse_args()

    # create output dir if necessary
    outdir = os.path.abspath(args.output)
    print(f"Output directory: {outdir}")

    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception as e:
        print(f"Fatal: Cannot create output directory {outdir}\n", e)
        sys.exit(1)

    # check access
    if not os.access(outdir, os.W_OK):
        print(f"Fatal: Cannot write to output directory {outdir}")
        sys.exit(1)

    # enter into output directory
    os.chdir(outdir)

    dataset_size = args.size

    if dataset_size == 'DEBUG':
        debug_archives = ['C_small_filename+path_sort_0GiB.tar.zstd_22',
                          'Python_small_filename+path_sort_0GiB.tar.zstd_22',
                          'random_small_filename+path_sort_0GiB.tar.zstd_22']

        debug_lists = ['C_small.csv', 'Python_small.csv', 'random_small.csv']

        # build the command line to be executed

        for archive, list in zip(debug_archives, debug_lists):
            try:
                link_compressed_archive = f"{ARCHIVES_LOCATION}/DEBUG/{archive}"
                link_list = f"{ARCHIVES_LOCATION}/DEBUG/{list}"

                # download the archives
                if args.v:
                    print(f"Downloading {link_compressed_archive}")
                    print(f"Downloading {link_list}")

                print(f"Downloading from the {ARCHIVES_LOCATION}")

                ok = 0
                command = f"wget --no-check-certificate {link_compressed_archive}"
                if exec_cmd_bool(command):
                    ok += 1

                command = f"wget --no-check-certificate {link_list}"
                if exec_cmd_bool(command):
                    ok += 1

                if ok == 2:
                    print("All archives successfully extracted!")
                else:
                    print(f"{2-ok} download failed! Check error messages.")

            except Exception as e:
                print(f"Fatal: Cannot download the DEBUG archives\n", e)
                sys.exit(1)

    elif dataset_size == '50GiB':
        try:
            # build the command line to be executed
            link_compressed_archive = f"{ARCHIVES_LOCATION}/50GiB_github/50GiB_github_filename_sort_50GiB.tar.zstd_22"
            link_compressed_archive_repositories = f"{ARCHIVES_LOCATION}/50GiB_github/repos_all_compressed.tar.zstd_22"
            link_list = f"{ARCHIVES_LOCATION}/50GiB_github/50GiB_github.csv"

            # download the archives
            if args.v:
                print(f"Downloading {link_compressed_archive}")
                print(
                    f"Downloading {link_compressed_archive_repositories}")
                print(f"Downloading {link_list}")

            print(f"Downloading from the {ARCHIVES_LOCATION}")

            ok = 0

            command = f"wget --no-check-certificate {link_compressed_archive}"
            if exec_cmd_bool(command):
                ok += 1

            command = f"wget --no-check-certificate {link_list}"
            if exec_cmd_bool(command):
                ok += 1

            command = f"wget --no-check-certificate {link_compressed_archive_repositories}"
            if exec_cmd_bool(command):
                ok += 1

            if ok == 3:
                print("All archives successfully downloaded!")
            else:
                print(f"{3-ok} download failed! Check error messages.")

        except Exception as e:
            print(f"Fatal: Cannot download the {dataset_size} archives\n", e)
            sys.exit(1)

    else:
        if 'all' in args.languages:
            args.languages = ['C', 'Python', 'Javascript', 'Java', 'random']

        ok = 0
        for dataset_language in args.languages:
            try:
                # build the command line to be executed
                link_compressed_archive = f"{ARCHIVES_LOCATION}/{dataset_size}/{dataset_language}_selection/{dataset_language}_selection_filename+path_sort_{dataset_size}.tar.zstd_22"
                link_list = f"{ARCHIVES_LOCATION}/{dataset_size}/{dataset_language}_selection/{dataset_language}_selection_{dataset_size}_info.csv"

                if args.v:
                    print(f"Downloading {link_compressed_archive}")
                    print(f"Downloading {link_list}")

                print(f"Downloading from the {ARCHIVES_LOCATION}")
                command = f"wget --no-check-certificate {link_compressed_archive}"
                if exec_cmd_bool(command):
                    ok += 1

                command = f"wget --no-check-certificate {link_list}"
                if exec_cmd_bool(command):
                    ok += 1

            except Exception as e:
                print(
                    f"Fatal: Cannot download the {dataset_size} archives\n", e)
                sys.exit(1)

        if ok == 2 * len(args.languages):
            print("All archives successfully downloaded!")
        else:
            if ok == len(args.languages):
                print("Just the lists are successfully downloaded (some archives migth not be available)!")
            print(
                f"{2*len(args.languages)-ok} download failed! Check error messages.")

    sys.exit(0)
