#!/usr/bin/env python3
import os
import sys
import time
import argparse
import getpass
from shutil import which
import pandas as pd
import numpy as np
from generate_tar_archive import compress_decompress_from_df
from utils import *

if sys.version_info < (3, 9):
    from utils_mmap import ssdeep_sort_mmap, hybrid_type_new_mmap

Description = """
Permute-Partition-Compress paradigm on large file collections

Take as input a list of files (csv-file parameters), permute them 
according to one or more techniques (-p option), concatenate them and 
optionally split the concatenation in blocks (-b option), and finally
compress each block using one or more compressors (-c option).

The input files must be in the same directory (-i option). Temporary files
and compressed archives are stored in a user-provided directory (-o option)

Finally the archives are decompressed; the compression ratio and compression 
and decompression speed are reported on stdin.

"""

# absolute path of input and output directores
DEFAULT_INPUT_DIR = "/data/swh/blobs_uncompressed"
DEFAULT_OUTPUT_DIR = "/extralocal/swh/"

# Instantiate the parser
parser = argparse.ArgumentParser(
    description=Description, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('csv_file', metavar="csv-file", nargs='+',  # default=[os.path.join(REPO_DIR, 'examples/C_small.csv')],
                    help='List of files to compress (in csv format)')

parser.add_argument('-c', '--compressor', nargs='+', default=['zstd'],
                    # (workaround here https://www.gnu.org/software/tar/manual/html_node/gzip.html)')
                    help='Compressors to apply to each block, default: zstd\n'
                         'See doc for how to pass options to a compressor')

parser.add_argument('-p', '--permuter', nargs='+', default=['filename'],
                    choices=[
                        'random', 'list', 'filename', 'filename-path', 'tlshsort', 'ssdeepsort', 'simhashsort',
                        'minhashgraph', 'typemagika', 'typeminhashgraph', 'typemagikaminhashgraph', 'lengthsort',
                        'typemagikatlshsort', 'icf_tlsh_single_sort', 'icf_single_sort', 'icf_multi_sort',
                        'cf_single_sort', 'ocf_single_sort', 'oicf_single_sort', 'all'
                    ],
                    help='Permutation strategies, one or more of the following:\n'
                    '* random: Permute blobs randomly\n'
                    '* lengthsort: Sort blobs according to legth\n'
                    '* list: No permutation, just use the order in the csv list\n'
                    '* filename: Sort blobs according to filename\n'
                    '* filename-path: Sort blobs by filename and path\n'
                    '* tlshsort: Sort blobs by TLSH\n'
                    '* ssdeepsort: Sort blobs by ssdeep\n'
                    '* simhashsort: Sort blobs by simhash\n'
                    '* minhashgraph: Sort blobs by minhash graph\n'
                    '* typeminhashgraph: Group by type(mime+lang)\n'
                    '  and then by minhash-graph on the individual groups\n'
                    '* typemagika: Group by type(magika library)\n'
                    '* typemagikaminhashgraph: Group by type(magika library) and apply minhash graph to the groups\n'
                    '  and then by minhash-graph on the individual groups\n'
                    '* all: Run all the permuting algorithms above', metavar='PERM')

parser.add_argument('-b', '--block-size', nargs='+', default=['0'],
                    help='If 0 a single archive is created. Otherwise, blocks\n'
                    'of BLOCK_SIZE bytes are created before compression.\n'
                    'BLOCK_SIZE must be an integer followed by an unit\n'
                    'denoting a power of 1024. Examples: -b 512KiB -b 1MiB\n'
                    'Valid units are: KiB, MiB, GiB. Default: 0\n')

parser.add_argument('-i', '--input-dir', default=DEFAULT_INPUT_DIR,
                    help='Directory where the uncompressed blobs are stored'
                    f'default: {DEFAULT_INPUT_DIR}')

parser.add_argument('-o', '--output-dir', default=DEFAULT_OUTPUT_DIR,
                    help='Directory for temporary files and compressed archives'
                    f'default: {DEFAULT_OUTPUT_DIR}')

parser.add_argument('-k', '--keep-tar', action='store_true',
                    help='Keep tar archives after benchmark. The resulting\n'
                    'tar archives are stored in the `--output-dir` directory',
                    default=False)

parser.add_argument('-m', '--mmap', action='store_true',
                    help='Use mmap on data. The blobs must be concatenated in a single `*_big_archive` file\n'
                    'See the function `create_big_archive` in mmap_on_compressed_data.py',
                    default=False)

parser.add_argument('-s', '--stats', action='store_true',
                    help='Just print stats of the dataset, no benchmark is performed',
                    default=False)

parser.add_argument('--type-stats', action='store_true',
                    help='Print stats about the type of the blobs of the dataset, no benchmark is performed',
                    default=False)

parser.add_argument('-T', '--num-thread', default=16, type=int,
                    help='Number of thread used for the compress blocks in parallell, default: 16')

parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print verbose output', default=False)

parser.add_argument('-V', '--version', action='version',
                    help='Print version and exit',
                    version='%(prog)s 1.0')

args = parser.parse_args()

if len(args.csv_file) == 0 or len(args.compressor) == 0 or len(args.permuter) == 0:
    parser.print_help()
    print('Error: You must specify at least one csv file and one compressor and one ordering technique')
    # print('DEBUG: setting ordering technique to hybrid_sorted')
    # args.minhashgraph = True
    exit(1)


def from_block_size_to_bytes(block_size):
    if block_size == '0':
        return 0
    else:
        # the last 3 chars are the unit
        if len(block_size) < 3:
            print(
                'Error: block size must be an integer followed by a unit (KiB, MiB, GiB)')
            exit(1)
        if block_size[-3:] == 'KiB':
            return int(block_size[:-3]) * 1024
        elif block_size[-3:] == 'MiB':
            return int(block_size[:-3]) * 1024 * 1024
        elif block_size[-3:] == 'GiB':
            return int(block_size[:-3]) * 1024 * 1024 * 1024
        else:
            print(
                'Error: block size must be an integer followed by a unit (KiB, MiB, GiB)')
            exit(1)


if __name__ == "__main__":

    print('# Start: {}. Machine: {}. User: {}. Taking files from {}. Saving archives to {}. PID {}.'
          .format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                  str(os.uname()[1]),
                  getpass.getuser(),
                  str(args.input_dir),
                  str(args.output_dir),
                  os.getpid()),
          flush=True)

    keep = args.keep_tar

    input_dir = args.input_dir
    output_dir = args.output_dir

    # todo change in if and exit(1)
    if not os.path.isdir(input_dir):
        print(f"Fatal: missing input directory: {input_dir}")
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f"Fatal: missing output directory: {output_dir}")
        sys.exit(1)

    # total, used, free = shutil.disk_usage(args.output_dir)

    NUM_THREAD = int(args.num_thread)

    for dataset in args.csv_file:
        dataset_name = os.path.basename(dataset)

        if args.mmap:
            df = pd.read_csv(dataset,
                             dtype={'swhid': 'string', 'file_id': 'string', 'length': 'Int64',
                                    'local_path': 'string', 'filename': 'string', 'filepath': 'string',
                                    'byte_pointer': 'Int64'},
                             # usecols=['file_id', 'length', 'local_path', 'filename', 'filepath'],
                             on_bad_lines='skip',
                             engine='python',
                             encoding_errors='ignore')
        else:
            df = pd.read_csv(dataset,
                             dtype={'swhid': 'string', 'file_id': 'string', 'length': 'Int64',
                                    'local_path': 'string', 'filename': 'string', 'filepath': 'string'},
                             # usecols=['file_id', 'length', 'local_path', 'filename', 'filepath'],
                             on_bad_lines='skip',
                             engine='python',
                             encoding_errors='ignore')

        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        # print(df.head())

        dataset_name = dataset_name.replace('.csv', '')
        dataset_name = dataset_name.replace('_info', '')

        if args.stats:
            print(
                'DATASET,NUM_BLOBS,TOTAL_SIZE(GiB),AVG_BLOB_SIZE(KiB),MEDIAN_BLOB_SIZE(KiB),COMMIT_HASH,NOTES')
            print(
                f"{dataset_name},{len(df.index)},{round(df['length'].sum() / float_1GiB, 2)},"
                f"{round(df['length'].mean() / float_1KiB, 2)},{round(df['length'].median() / float_1KiB, 2)},"
                f"{repo.head.object.hexsha[:7]},just_stats", flush=True)
            sys.exit(0)

        if args.type_stats:
            from magika import Magika
            m = Magika()

            def guess_fun_magika_from_bytes(x):
                return m.identify_bytes(open(x, "rb").read(4096)).output.ct_label

            stats_from_filenames = get_stats_from_filename(df)
            if sys.version_info < (3, 9):
                stats_from_mimeguesslang = get_stats_from_type(
                    df, guess_fun_guesslang, input_dir)
            stats_from_magika = get_stats_from_type(
                df, guess_fun_magika_from_bytes, input_dir)

            print(
                'DATASET,NUM_BLOBS,TOTAL_SIZE(GiB),AVG_BLOB_SIZE(KiB),MEDIAN_BLOB_SIZE(KiB)')
            print(f"{dataset_name},{len(df.index)},{round(df['length'].sum() / float_1GiB, 2)},"
                  f"{round(df['length'].mean() / float_1KiB, 2)},{round(df['length'].median() / float_1KiB, 2)}", flush=True)

            print_stats(stats_from_filenames, 'stats_from_filenames')
            if sys.version_info < (3, 9):
                print_stats(stats_from_mimeguesslang,
                            'stats_from_mimeguesslang')
            print_stats(stats_from_magika, 'stats_from_magika')

            sys.exit(0)

        if args.block_size == ['0']:
            print('DATASET,NUM_BLOBS,TOTAL_SIZE(GiB),AVG_BLOB_SIZE(KiB),MEDIAN_BLOB_SIZE(KiB),TECHNIQUE,COMPRESSION_RATIO(%),ORDERING_TIME(s),COMPRESSION_TIME(s),COMPRESSION_SPEED(MiB/s),DECOMPRESSION_SPEED(MiB/s),COMMIT_HASH({}),NOTES'.format(
                repo.head.object.hexsha[:7]), flush=True)
        else:
            print("DATASET,NUM_BLOBS,TOTAL_SIZE(GiB),AVG_BLOB_SIZE(KiB),MEDIAN_BLOB_SIZE(KiB),TECHNIQUE,COMPRESSION_RATIO(%),ORDERING_TIME(s),COMPRESSION_TIME(s),COMPRESSION_SPEED(MiB/s),FULL_DECOMPRESSION_SPEED(MiB/s),TIME_BLOB_DECOMPRESSION(ms),THROUGHPUT(blobs/s),COMMIT_HASH({}),NOTES".format(
                repo.head.object.hexsha[:7]), flush=True)

        if 'all' in args.permuter:
            if sys.version_info < (3, 9):
                #args.permuter = ['random', 'list', 'lengthsort', 'filename', 'filename-path', 'simhashsort',
                #                'tlshsort', 'ssdeepsort', 'minhashgraph', 'typemagika', 'typeminhashgraph', 'typemagikaminhashgraph']
                                args.permuter = ['random', 'filename', 'filename-path', 'simhashsort',
                                'tlshsort', 'minhashgraph', 'typemagika', 'typemagikatlshsort', 'typemagikaminhashgraph']
            else:
                args.permuter = ['random', 'list', 'lengthsort', 'filename', 'filename-path', 'simhashsort',
                                'tlshsort', 'minhashgraph', 'typemagika', 'typemagikaminhashgraph']

        for compressor in args.compressor:
            # the compressor script must exist
            assert (os.path.isfile(compressor) or which(compressor) is not None)

            if not os.access(dataset, os.F_OK):
                print(f"Fatal: missing input file: {dataset}")
                sys.exit(1)
            if not os.access(dataset, os.R_OK):
                print(f"Fatal: Cannot read input file: {dataset}")
                sys.exit(1)

            for permuter in args.permuter:

                pd.set_option('display.max_columns', None)

                # print(df.head())

                # if free < df['length'].sum() / 4:
                #     print("Probably not enough space on disk to run the benchmark")

                if permuter == 'random':
                    np.random.seed(42)
                    random_permutation = np.random.permutation(len(df.index))
                    sorting_time = 0
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(random_permutation,
                                                    'random_order', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)
                if permuter == 'list':
                    # just take the order in which the blobs are listed in the dataframe
                    sorting_time = 0
                    permutation = range(len(df.index))
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(permutation,
                                                    'list_order', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)
                if permuter == 'filename':
                    start_time = time.time()
                    # reverse all filenames, to get <extension(type)>.<name>
                    df['filename'] = df['filename'].str[::-1]
                    # doesn't modify the dataframe, just returns the index
                    ordered_rows = df.sort_values(['filename', 'length'], ascending=[
                        True, False]).index
                    sorting_time = time.time() - start_time
                    # set it back to the original
                    df['filename'] = df['filename'].str[::-1]
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(ordered_rows,
                                                    'filename_sort', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)

                if permuter == 'filename-path':
                    start_time = time.time()
                    # reverse all filepath, to get <extension(type)>.<name>.<dir>...
                    df['filepath'] = df['filepath'].str[::-1]
                    # doesn't modify the dataframe, just returns the index
                    ordered_rows = df.sort_values(['filepath', 'length'], ascending=[
                        True, False]).index
                    sorting_time = time.time() - start_time
                    # set it back to the original
                    df['filepath'] = df['filepath'].str[::-1]
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(ordered_rows,
                                                    'filename+path_sort', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)

                if permuter == 'lengthsort':
                    start_time = time.time()
                    # doesn't modify the dataframe, just returns the index
                    ordered_rows = df.sort_values(
                        ['length'], ascending=[False]).index
                    sorting_time = time.time() - start_time
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(ordered_rows,
                                                    'length_sort', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)

                if permuter == 'simhashsort':
                    start_time = time.time()
                    ordered_rows = simhash_sort(
                        df, 1, 256, len_limit, input_dir)
                    sorting_time = time.time() - start_time
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(ordered_rows,
                                                    'simhash_sort', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)
                if permuter == 'tlshsort':
                    start_time = time.time()
                    ordered_rows = TLSH_sort(df, input_dir)
                    sorting_time = time.time() - start_time
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(ordered_rows,
                                                    'TLSH_sort', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)

                if permuter == 'ssdeepsort':
                    if sys.version_info < (3, 9):
                        start_time = time.time()
                        ordered_rows = ssdeep_sort(df, input_dir)
                        sorting_time = time.time() - start_time
                        for arg_block_size in args.block_size:
                            block_size_in_bytes = from_block_size_to_bytes(
                                arg_block_size)
                            compress_decompress_from_df(ordered_rows,
                                                        'ssdeep_sort', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)
                    else:
                        print(
                            "ssdeep_sort is not supported with python > 3.8. Please use tlshsort instead")

                if permuter == 'typeminhashgraph':
                    if sys.version_info < (3, 9):
                        from guesslang import Guess
                        import magic
                        # loading the model is very time consuming, so it is done only if the technique is actually used
                        guess = Guess()

                        def guess_fun_from_content(x): return magic.from_buffer(
                            open(x, "rb").read(2048), mime=True)

                        def guess_fun_guesslang(x):
                            # read just 10K of the files
                            # checks on the size are made before
                            file_content = read_file_size(x, 10*(2**10))
                            # assert(len(file_content) > 0)
                            # file with just \n, \t, or white spaces
                            if not file_content.strip():
                                return 'too_small'
                            return guess.language_name(file_content)

                        def guess_fun_guesslang_content(file_content):
                            if not file_content.strip():
                                return 'too_small'
                            return guess.language_name(file_content)

                        def guess_fun_from_header(x):
                            return magic.from_file(x, mime=True)

                        def guess_fun_from_header_content(x):
                            return magic.from_buffer(x, mime=True)

                        start_time = time.time()
                        ordered_rows = hybrid_type_new(
                            df, guess_fun_from_header, 'text', guess_fun_guesslang, row_minhashgraph_unionfind_fun, input_dir)

                        sorting_time = time.time() - start_time
                        for arg_block_size in args.block_size:
                            block_size_in_bytes = from_block_size_to_bytes(
                                arg_block_size)
                            compress_decompress_from_df(ordered_rows,
                                                        'typeminhashgraph', dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)
                    else:
                        print(
                            "typeminhashgraph is not supported with python > 3.8. Please use typemagikaminhashgraph instead")

                if permuter == 'minhashgraph':
                    for f in [256]:
                        for r in [64]:
                            start_time = time.time()
                            ordered_rows = minhash_graph_technique_unionfind(
                                df, 1, f, r, len_limit, input_dir)
                            sorting_time = time.time() - start_time
                            for arg_block_size in args.block_size:
                                block_size_in_bytes = from_block_size_to_bytes(
                                    arg_block_size)
                                compress_decompress_from_df(ordered_rows,
                                                            'minhash_graph_tlshsort_uf_f{}_r{}'.format(f, r), dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, f'f{f}_r{r}', args)

                if permuter == 'typemagika' or permuter == 'typemagikatlshsort' or permuter == 'typemagikaminhashgraph':
                    # test to solve 2024-03-16 11:09:18.079207986 [E:onnxruntime:Default, env.cc:254 ThreadMain] pthread_setaffinity_np failed
                    # os.environ["OMP_NUM_THREADS"]='1'

                    from magika import Magika
                    # loading the model is very time consuming, so it is done only if the technique is actually used
                    m = Magika()

                    def guess_fun_magika_from_bytes(x):
                        # 4096 is memory page size
                        return m.identify_bytes(open(x, "rb").read(4096)).output.ct_label

                    # To make work magika with the path. TODO: to test
                    def guess_fun_magika_from_path(x):
                        return m.identify_path(x).output.ct_label

                if permuter == 'typemagika':
                    start_time = time.time()
                    ordered_rows = hybrid_type_1guess(
                        df, guess_fun_magika_from_bytes, None, input_dir)
                    technique_name = 'typemagika'

                    sorting_time = time.time() - start_time
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(ordered_rows,
                                                    technique_name, dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)

                if permuter == 'typemagikatlshsort':
                    start_time = time.time()
                    ordered_rows = hybrid_type_1guess(
                        df, guess_fun_magika_from_bytes, tlsh_sort_list, input_dir)
                    technique_name = 'typemagikatlshsort'

                    sorting_time = time.time() - start_time
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(ordered_rows,
                                                    technique_name, dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)


                if permuter == 'typemagikaminhashgraph':
                    start_time = time.time()

                    ordered_rows = hybrid_type_1guess(
                        df, guess_fun_magika_from_bytes, row_minhashgraph_unionfind_fun, input_dir)
                    technique_name = 'typemagikaminhashgraph'

                    sorting_time = time.time() - start_time
                    for arg_block_size in args.block_size:
                        block_size_in_bytes = from_block_size_to_bytes(
                            arg_block_size)
                        compress_decompress_from_df(ordered_rows,
                                                    technique_name, dataset_name, df, compressor, sorting_time, block_size_in_bytes, arg_block_size, 'None', args)

                if permuter == 'icf-tlsh':
                    start_time = time.time()
                    ordered_rows = TLSH_names_sort(df, input_dir)
                    sorting_time = time.time() - start_time

                    for arg_block_size in args.block_size:
                        compress_decompress_from_df(
                            ordered_rows, permuter, dataset_name, df, compressor, sorting_time,
                            from_block_size_to_bytes(arg_block_size), arg_block_size, '', args)

                if permuter == 'icf_tlsh_single_sort':
                    start_time = time.time()
                    ordered_rows = icf_tlsh_single_sort(df, input_dir)
                    sorting_time = time.time() - start_time

                    for arg_block_size in args.block_size:
                        compress_decompress_from_df(
                            ordered_rows, permuter, dataset_name, df, compressor, sorting_time,
                            from_block_size_to_bytes(arg_block_size), arg_block_size, '', args)

                if permuter == 'icf_single_sort':
                    start_time = time.time()
                    ordered_rows = icf_single_sort(df, input_dir)
                    sorting_time = time.time() - start_time

                    for arg_block_size in args.block_size:
                        compress_decompress_from_df(
                            ordered_rows, permuter, dataset_name, df, compressor, sorting_time,
                            from_block_size_to_bytes(arg_block_size), arg_block_size, '', args)

                if permuter == 'icf_multi_sort':
                    start_time = time.time()
                    ordered_rows = icf_multi_sort(df, input_dir)
                    sorting_time = time.time() - start_time

                    for arg_block_size in args.block_size:
                        compress_decompress_from_df(
                            ordered_rows, permuter, dataset_name, df, compressor, sorting_time,
                            from_block_size_to_bytes(arg_block_size), arg_block_size, '', args)

                if permuter == 'cf_single_sort':
                    start_time = time.time()
                    ordered_rows = cf_single_sort(df, input_dir)
                    sorting_time = time.time() - start_time

                    for arg_block_size in args.block_size:
                        compress_decompress_from_df(
                            ordered_rows, permuter, dataset_name, df, compressor, sorting_time,
                            from_block_size_to_bytes(arg_block_size), arg_block_size, '', args)

                if permuter == 'ocf_single_sort':
                    start_time = time.time()
                    ordered_rows = ocf_single_sort(df, input_dir)
                    sorting_time = time.time() - start_time

                    for arg_block_size in args.block_size:
                        compress_decompress_from_df(
                            ordered_rows, permuter, dataset_name, df, compressor, sorting_time,
                            from_block_size_to_bytes(arg_block_size), arg_block_size, '', args)

                if permuter == 'oicf_single_sort':
                    start_time = time.time()
                    ordered_rows = oicf_single_sort(df, input_dir)
                    sorting_time = time.time() - start_time

                    for arg_block_size in args.block_size:
                        compress_decompress_from_df(
                            ordered_rows, permuter, dataset_name, df, compressor, sorting_time,
                            from_block_size_to_bytes(arg_block_size), arg_block_size, '', args)

    print('')
    print("# Ending time : ", time.strftime(
        '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    sys.exit(0)
