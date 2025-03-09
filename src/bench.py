#!/usr/bin/env python3
import argparse
import getpass
import os
import textwrap
from pathlib import Path
from shutil import which
import time

import git
import pandas as pd

from generate_tar_archive import compress_decompress_from_df
from permute.permuters import Permuters


# TODO: cf-tlsh
# TODO: icf-tlsh
# TODO: cf-minhash
# TODO: icf-minhash


# Instantiate the parser
def build_parser():
    # Absolute path of input and output directories
    DEFAULT_INPUT_DIR = "/data/swh/blobs_uncompressed"
    DEFAULT_OUTPUT_DIR = "/extralocal/swh/"

    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            Permute-Partition-Compress paradigm on large file collections

            Take as input a list of files (csv-file parameters), permute them
            according to one or more techniques (-p option), concatenate them and
            optionally split the concatenation in blocks (-b option), and finally
            compress each block using one or more compressors (-c option).

            The input files must be in the same directory (-i option). Temporary files
            and compressed archives are stored in a user-provided directory (-o option)

            Finally the archives are decompressed; the compression ratio and compression
            and decompression speed are reported on stdin.
        """),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'csv_file', metavar="csv-file", nargs='+',
        # default=[os.path.join(REPO_DIR, 'examples/C_small.csv')],
        help='List of files to compress (in csv format)')

    parser.add_argument(
        '--limit', default=0,
        help='Limit input csv files to given limit')

    parser.add_argument(
        '-c', '--compressor', nargs='+', default=['zstd'],
        # (workaround here https://www.gnu.org/software/tar/manual/html_node/gzip.html)')
        help='Compressors to apply to each block, default: zstd\n'
             'See doc for how to pass options to a compressor')

    parser.add_argument(
        '-p', '--permuter', nargs='+', default=['filename'],
        choices=Permuters.choices(),
        # [
        #     'random', 'list', 'filename', 'filename-path', 'tlshsort', 'ssdeepsort', 'simhashsort',
        #     'minhashgraph', 'typemagika', 'typeminhashgraph', 'typemagikaminhashgraph', 'lengthsort',
        #     'typemagikatlshsort', 'icf_tlsh_single_sort', 'icf_single_sort', 'icf_multi_sort',
        #     'cf_single_sort', 'ocf_single_sort', 'oicf_single_sort', 'all'
        # ]
        help=Permuters.help(),
        # help='Permutation strategies, one or more of the following:\n'
        #      '* random: Permute blobs randomly\n'
        #      '* lengthsort: Sort blobs according to legth\n'
        #      '* list: No permutation, just use the order in the csv list\n'
        #      '* filename: Sort blobs according to filename\n'
        #      '* filename-path: Sort blobs by filename and path\n'
        #      '* tlshsort: Sort blobs by TLSH\n'
        #      '* ssdeepsort: Sort blobs by ssdeep\n'
        #      '* simhashsort: Sort blobs by simhash\n'
        #      '* minhashgraph: Sort blobs by minhash graph\n'
        #      '* typeminhashgraph: Group by type(mime+lang)\n'
        #      '  and then by minhash-graph on the individual groups\n'
        #      '* typemagika: Group by type(magika library)\n'
        #      '* typemagikaminhashgraph: Group by type(magika library) and apply minhash graph to the groups\n'
        #      '  and then by minhash-graph on the individual groups\n'
        #      '* all: Run all the permuting algorithms above',
        metavar='PERM')

    parser.add_argument(
        '-b', '--block-size', nargs='+', default=['0'],
        help='If 0 a single archive is created. Otherwise, blocks\n'
             'of BLOCK_SIZE bytes are created before compression.\n'
             'BLOCK_SIZE must be an integer followed by an unit\n'
             'denoting a power of 1024. Examples: -b 512KiB -b 1MiB\n'
             'Valid units are: KiB, MiB, GiB. Default: 0\n')

    parser.add_argument(
        '-i', '--input-dir', default=DEFAULT_INPUT_DIR,
        help='Directory where the uncompressed blobs are stored'
             f'default: {DEFAULT_INPUT_DIR}')

    parser.add_argument(
        '-o', '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help='Directory for temporary files and compressed archives'
             f'default: {DEFAULT_OUTPUT_DIR}')

    parser.add_argument(
        '-k', '--keep-tar', action='store_true',
        help='Keep tar archives after benchmark. The resulting\n'
             'tar archives are stored in the `--output-dir` directory',
        default=False)

    parser.add_argument(
        '-m', '--mmap', action='store_true',
        help='Use mmap on data. The blobs must be concatenated in a single `*_big_archive` file\n'
             'See the function `create_big_archive` in mmap_on_compressed_data.py',
        default=False)

    parser.add_argument(
        '-s', '--stats', action='store_true',
        help='Just print stats of the dataset, no benchmark is performed',
        default=False)

    parser.add_argument(
        '--type-stats', action='store_true',
        help='Print stats about the type of the blobs of the dataset, no benchmark is performed',
        default=False)

    parser.add_argument(
        '-T', '--num-thread', default=16, type=int,
        help='Number of thread used for the compress blocks in parallel, default: 16')

    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print verbose output', default=False)

    parser.add_argument(
        '-V', '--version', action='version',
        help='Print version and exit',
        version='%(prog)s 1.0')

    return parser


def validate_args(parser, args):
    if len(args.csv_file) == 0 or len(args.compressor) == 0 or len(args.permuter) == 0:
        parser.print_help()
        print()
        print('Error: You must specify at least one csv file and one compressor and one ordering technique')
        exit(1)

    args.limit = int(args.limit)

    for i, file in enumerate(args.csv_file):
        try:
            file = args.csv_file[i] = Path(file).resolve(strict=True)
        except FileNotFoundError:
            print(f"Fatal: cannot find csv_file: {file!s}")
            exit(1)

        if not os.access(file, os.R_OK):
            print(f"Fatal: Cannot read input file: {file.name}")
            exit(1)

    args.input_dir = Path(args.input_dir).resolve(strict=True)
    if not os.path.isdir(args.input_dir):
        print(f"Fatal: missing input directory: {args.input_dir}")
        exit(1)

    args.output_dir = Path(args.output_dir).resolve(strict=True)
    if not os.path.isdir(args.output_dir):
        print(f"Fatal: missing output directory: {args.output_dir}")
        exit(1)

    for i, compressor in enumerate(args.compressor):
        try:
            args.compressor[i] = Path(which(compressor)).resolve(strict=True)
        except FileNotFoundError:
            print(f"Fatal: cannot find compressor: {compressor}")
            exit(1)

    def validate_block_sizes():
        for block_size in args.block_size:
            if block_size == '0':
                block_byte_size = 0
            else:
                print(block_size)
                # the last 3 chars are the unit
                if len(block_size) < 3:
                    print('Error: block size must be an integer followed by a unit (KiB, MiB, GiB)')
                    exit(1)

                if block_size[-3:] == 'KiB':
                    block_byte_size = int(block_size[:-3]) * 1024
                elif block_size[-3:] == 'MiB':
                    block_byte_size = int(block_size[:-3]) * 1024 * 1024
                elif block_size[-3:] == 'GiB':
                    block_byte_size = int(block_size[:-3]) * 1024 * 1024 * 1024
                else:
                    print('Error: block size must be an integer followed by a unit (KiB, MiB, GiB)')
                    exit(1)
            yield block_byte_size

    args.block_byte_size = [*validate_block_sizes()]

    return args


def main():
    repo = git.Repo(search_parent_directories=True)

    parser = build_parser()
    args = parser.parse_args()
    args = validate_args(parser, args)

    print(
        f'# Start: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}\n'
        f'  Machine: {os.uname()[1]!s}\n'
        f'  User: {getpass.getuser()}\n'
        f'  PID: {os.getpid()}\n'
        f'  Taking files from {args.input_dir!s}\n'
        f'  Saving archives to {args.output_dir!s}\n'
    )

    # TODO
    # NUM_THREAD = int(args.num_thread)

    pd.set_option('display.max_columns', None)

    for dataset in args.csv_file:
        # total, used, free = shutil.disk_usage(args.output_dir)
        # if free < df['length'].sum() / 4:
        #     print("Probably not enough space on disk to run the benchmark")

        if args.mmap:
            df = pd.read_csv(
                dataset,
                dtype={
                    'swhid': 'string', 'file_id': 'string', 'length': 'Int64',
                    'local_path': 'string', 'filename': 'string', 'filepath': 'string',
                    'byte_pointer': 'Int64'
                },
                # usecols=['file_id', 'length', 'local_path', 'filename', 'filepath'],
                on_bad_lines='skip',
                engine='python',
                encoding_errors='ignore',
                nrows=args.limit if args.limit > 0 else None
            )
        else:
            df = pd.read_csv(
                dataset,
                dtype={
                    'swhid': 'string', 'file_id': 'string', 'length': 'Int64',
                    'local_path': 'string', 'filename': 'string', 'filepath': 'string'
                },
                # usecols=['file_id', 'length', 'local_path', 'filename', 'filepath'],
                on_bad_lines='skip',
                engine='python',
                encoding_errors='ignore',
                nrows=args.limit if args.limit > 0 else None
            )

        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        # print(df.head())

        dataset_name = dataset.stem.replace('_info', '')

        if args.stats:
            command_stats(dataset_name, df, repo)

        if args.type_stats:
            command_type_stats(dataset_name, df)

        if args.block_size == ['0']:
            print(
                'DATASET,NUM_BLOBS,TOTAL_SIZE(GiB),AVG_BLOB_SIZE(KiB),MEDIAN_BLOB_SIZE(KiB),TECHNIQUE,COMPRESSION_RATIO(%),ORDERING_TIME(s),COMPRESSION_TIME(s),COMPRESSION_SPEED(MiB/s),DECOMPRESSION_SPEED(MiB/s),COMMIT_HASH({}),NOTES'.format(
                    repo.head.object.hexsha[:7]), flush=True)
        else:
            print(
                "DATASET,NUM_BLOBS,TOTAL_SIZE(GiB),AVG_BLOB_SIZE(KiB),MEDIAN_BLOB_SIZE(KiB),TECHNIQUE,COMPRESSION_RATIO(%),ORDERING_TIME(s),COMPRESSION_TIME(s),COMPRESSION_SPEED(MiB/s),FULL_DECOMPRESSION_SPEED(MiB/s),TIME_BLOB_DECOMPRESSION(ms),THROUGHPUT(blobs/s),COMMIT_HASH({}),NOTES".format(
                    repo.head.object.hexsha[:7]), flush=True)

        for compressor in args.compressor:
            for permuter in args.permuter:
                sorting_time, ordered_rows = Permuters[permuter.upper()].value.permute_and_measure(
                    df, args.input_dir, args.num_thread
                )
                for block_size, block_byte_size in zip(args.block_size, args.block_byte_size):
                    compress_decompress_from_df(
                        ordered_rows, permuter, dataset_name, df, compressor, sorting_time,
                        block_byte_size, block_size, repo, '', args
                    )

    print()
    print("# Ending time : ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    exit(0)


def command_type_stats(dataset_name, df):
    from magika import Magika
    m = Magika()

    def guess_fun_magika_from_bytes(x):
        return m.identify_bytes(open(x, "rb").read(4096)).output.ct_label

    stats_from_filenames = get_stats_from_filename(df)
    if sys.version_info < (3, 9):
        stats_from_mimeguesslang = get_stats_from_type(df, guess_fun_guesslang, input_dir)
    stats_from_magika = get_stats_from_type(df, guess_fun_magika_from_bytes, input_dir)
    print('DATASET,NUM_BLOBS,TOTAL_SIZE(GiB),AVG_BLOB_SIZE(KiB),MEDIAN_BLOB_SIZE(KiB)')
    print(
        f"{dataset_name},{len(df.index)},{round(df['length'].sum() / float_1GiB, 2)},"
        f"{round(df['length'].mean() / float_1KiB, 2)},{round(df['length'].median() / float_1KiB, 2)}"
    )
    print_stats(stats_from_filenames, 'stats_from_filenames')
    if sys.version_info < (3, 9):
        print_stats(stats_from_mimeguesslang, 'stats_from_mimeguesslang')
    print_stats(stats_from_magika, 'stats_from_magika')
    exit(0)


def command_stats(dataset_name, df, repo):
    print('DATASET,NUM_BLOBS,TOTAL_SIZE(GiB),AVG_BLOB_SIZE(KiB),MEDIAN_BLOB_SIZE(KiB),COMMIT_HASH,NOTES')
    print(
        f"{dataset_name},{len(df.index)},{round(df['length'].sum() / float_1GiB, 2)},"
        f"{round(df['length'].mean() / float_1KiB, 2)},{round(df['length'].median() / float_1KiB, 2)},"
        f"{repo.head.object.hexsha[:7]},just_stats"
    )
    exit(0)


if __name__ == "__main__":
    main()
