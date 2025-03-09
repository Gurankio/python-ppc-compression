import numbers
import os
import shutil
import sys
import time
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from sys import platform

from utils.generic import float_1KiB, float_1MiB, float_1GiB, exec_cmd, check_is_permutation, write_filenames_to_file


def generated_filename(dataset_name, chosen_compressor_name, curr_block_idx, block_size, technique_name,
                       total_uncompressed_size_GiB):
    to_return = '{}_{}_{}GiB_block_compressed_{}.tar.{}'.format(
        dataset_name, technique_name, str(round(total_uncompressed_size_GiB)), str(block_size), chosen_compressor_name)
    to_return = '{0:09d}_'.format(curr_block_idx) + to_return
    return to_return


def compress_decompress_from_df(ordered_rows, technique_name, dataset_name, df, chosen_compressor, sorting_time,
                                block_size_bytes, block_size_str, repo, notes='None', args=None):
    if block_size_bytes == 0:
        compress_decompress_single_from_df(ordered_rows, technique_name, dataset_name,
                                           df, chosen_compressor, sorting_time, repo, notes, args)
    else:
        compress_decompress_blocks_from_df(ordered_rows, technique_name, dataset_name, df, chosen_compressor,
                                           sorting_time, block_size_bytes, block_size_str, repo, notes, args)


def compress_decompress_single_from_df(ordered_rows, technique_name, dataset_name, df, chosen_compressor, sorting_time,
                                       repo, notes='None', args=None):
    assert (check_is_permutation(ordered_rows, len(df.index)))

    input_dir = args.input_dir
    output_dir = args.output_dir
    keep_tar = args.keep_tar

    total_uncompressed_size = df['length'].sum()

    avg_file_sizeKiB = df['length'].mean() / float_1KiB
    median_file_sizeKiB = df['length'].median() / float_1KiB

    total_uncompressed_size_MiB = total_uncompressed_size / float_1MiB
    total_uncompressed_size_GiB = total_uncompressed_size / float_1GiB

    str_size_GiB = str(int(round(total_uncompressed_size_GiB, 2)))

    work_name = 'tmp.SWH_storage_{}_{}_{}_{}_{}'.format(
        technique_name, chosen_compressor.stem, dataset_name, str_size_GiB, os.getpid()
    )
    work_path = output_dir / work_name

    try:
        work_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Fatal: Cannot create output directory {work_path}\n", e)
        exit(1)

    filenames = []
    for row in ordered_rows:
        assert (isinstance(row, numbers.Number))
        assert (row >= 0 and row < len(df.index))

        filename = Path(df.iloc[int(row)]['local_path']) / df.iloc[int(row)]['file_id']
        filenames.append(filename)

    list_files_path = work_path / 'list_files_compression.txt'
    list_files_path.write_text('\n'.join(map(str, filenames)))

    generated_name = '{}_{}_{}GiB.tar.{}'.format(dataset_name, technique_name, str_size_GiB, chosen_compressor.stem)
    generated_path = work_path / generated_name

    # Interacting with compressors flags, tar, and subprocess.Popen is a mess
    # Workaround: using script
    # Parallelism is archived by the compressor
    tar = 'gtar' if platform == 'darwin' else 'tar'
    to_exec = (f"{tar}"
               f" -cf {generated_path}"
               f" -C {input_dir}"
               f" -T {list_files_path}"
               f" -I{chosen_compressor}"
               f" --owner=0 --group=0 "
               f"--no-same-owner --no-same-permissions")

    start_time = time.time()
    exec_cmd(to_exec)
    compression_time = time.time() - start_time
    compressed_size = int(os.path.getsize(generated_path))

    list_files_path.unlink()

    start_time = time.time()
    exec_cmd(f"{tar} -C {work_path} -xf {generated_path} -I{chosen_compressor}")
    decompression_time = time.time() - start_time

    if keep_tar:
        # Copy generated file to the output directory
        print(f"# Generated file: {output_dir / generated_path.name}")
        shutil.copy(generated_path, output_dir / generated_path)

    try:
        generated_path.unlink()
        shutil.rmtree(work_path, ignore_errors=True)

        if platform == 'darwin':
            (work_path / '.DS_Store').unlink(missing_ok=True)
            work_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"Could not remove generated file.\n"
              f"{e}")

    # ('DATASET,NUM_FILES,TOTAL_SIZE(GiB),AVG_FILE_SIZE(KiB),MEDIAN_FILE_SIZE(KiB),TECHNIQUE,COMPRESSION_RATIO(%),ORDERING_TIME(s),COMPRESSION_TIME(s),COMPRESSION_SPEED(MiB/s),DECOMPRESSION_SPEED(MiB/s),COMMIT_HASH,NOTES')
    print(
        '{},{},{},{},{},{}+{},{},{},{},{},{},{},{}'.format(
            dataset_name,
            len(df.index),
            str(round(total_uncompressed_size_GiB, 2)),
            str(round(avg_file_sizeKiB, 2)),
            str(round(median_file_sizeKiB, 2)),
            technique_name,
            chosen_compressor.stem,
            str(round((compressed_size / total_uncompressed_size * 100), 2)),  # compression ratio
            str(round(sorting_time, 2)),  # sorting time
            str(round(compression_time, 2)),  # compression time
            str(round(total_uncompressed_size_MiB / \
                      float(compression_time + sorting_time), 2)),  # compression speed
            str(round(total_uncompressed_size_MiB / float(decompression_time), 2)),  # decompression speed
            repo.head.object.hexsha[:7],
            notes)
    )


# block_size is in bytes
def compress_decompress_blocks_from_df(ordered_rows, technique_name, dataset_name, df, chosen_compressor, sorting_time,
                                       block_size, block_size_str, repo, notes='None', args=None):
    assert (check_is_permutation(ordered_rows, len(df.index)))

    input_dir = args.input_dir
    output_dir = args.output_dir
    keep_tar = args.keep_tar
    NUM_THREAD = args.num_thread

    compressor_no_flags = os.path.basename(chosen_compressor.split()[0])

    path_working_dir = os.path.join(output_dir, 'tmp.SWH_random_access_{}_{}_{}_{}'.format(
        technique_name, compressor_no_flags, dataset_name, os.getpid()))

    # print(f'got {block_size} as block size')
    try:
        os.makedirs(path_working_dir, exist_ok=True)
    except Exception as e:
        print(f"Fatal: Cannot create output directory {path_working_dir}\n", e)
        sys.exit(1)

    compressed_size = 0
    compression_time = 0
    decompression_time = 0

    total_uncompressed_size = df['length'].sum()

    avg_file_sizeKiB = df['length'].mean() / float_1KiB
    median_file_sizeKiB = df['length'].median() / float_1KiB

    # total_uncompressed_size_KiB = total_uncompressed_size / float_1KiB
    total_uncompressed_size_MiB = total_uncompressed_size / float_1MiB
    total_uncompressed_size_GiB = total_uncompressed_size / float_1GiB

    os.chdir(path_working_dir)

    num_files = len(df.index)
    curr_block_idx = 0
    curr_block_size = 0
    curr_block_list = []

    filename_archive_map = []
    start_time = time.time()

    tar = 'gtar' if platform == "darwin" else 'tar'

    def compress_one_block(dataset_name, curr_block_idx, curr_block_list, path_working_dir, input_dir,
                           chosen_compressor, compressor_no_flags, block_size_str):
        # Interacting with compressors flags, tar, and subprocess.Popen is a mess
        # Workaround: using script
        idx_string = str(curr_block_idx).zfill(10)
        list_files_filename = os.path.join(
            path_working_dir, 'list_files_block_{}.txt.{}'.format(idx_string, compressor_no_flags))
        write_filenames_to_file(curr_block_list, list_files_filename)
        generated_file = generated_filename(
            dataset_name, compressor_no_flags, curr_block_idx, block_size_str, technique_name,
            total_uncompressed_size_GiB)
        to_exec = f"{tar} -cf {generated_file} -C {input_dir} -T {list_files_filename} -I{chosen_compressor} --owner=0 --group=0 --no-same-owner --no-same-permissions"
        exec_cmd(to_exec)

    print(f"NUM_THREAD: {NUM_THREAD}")

    with ThreadPoolExecutor(NUM_THREAD) as executor:
        for i, row in enumerate(ordered_rows):
            # this_file_size = os.path.getsize(df.iloc[int(row)]['sha1'])
            this_file_size = df.iloc[int(row)]['length']
            curr_block_size += this_file_size
            curr_block_list.append(os.path.join(
                df.iloc[int(row)]['local_path'], df.iloc[int(row)]['file_id']))

            filename_archive_map.append(
                [os.path.join(df.iloc[int(row)]['local_path'], df.iloc[int(row)]['file_id']), generated_filename(
                    dataset_name, compressor_no_flags, curr_block_idx, block_size_str, technique_name,
                    total_uncompressed_size_GiB)])

            # filename_archive_map.append([df.iloc[int(row)]['sha1'], generated_filename(
            #    dataset_name, compressor_no_flags, curr_block_idx, block_size_exponent)])

            if (curr_block_size >= block_size) or i == (num_files - 1):
                old_block_idx = curr_block_idx
                old_curr_block_list = curr_block_list

                # compress_one_block(dataset_name, old_block_idx, old_curr_block_list, path_working_dir, input_dir, chosen_compressor, compressor_no_flags, block_size_str)
                executor.submit(compress_one_block, dataset_name, old_block_idx, old_curr_block_list,
                                path_working_dir, input_dir, chosen_compressor, compressor_no_flags, block_size_str)
                curr_block_idx += 1
                curr_block_size = 0
                curr_block_list = []

    compression_time = time.time() - start_time

    # print("Compressed {} blocks".format(curr_block_idx))

    # computing compressed_size
    for i in range(0, curr_block_idx):
        generated_file = generated_filename(dataset_name,
                                            compressor_no_flags, i, block_size_str, technique_name,
                                            total_uncompressed_size_GiB)
        compressed_size += int(os.path.getsize(generated_file))

    start_time = time.time()

    percentage_decompression = 10
    divisor_decompression = float(percentage_decompression / 100)

    # Just decompress a small portion (sample) of the blocks (for time reasons)
    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def decompress_one_block(generated_file, chosen_compressor):
            exec_cmd(f"{tar} -xf {generated_file} -I{chosen_compressor}")

        # create a random permutation of int(curr_block_idx * divisor_decompression) taken from range(0, curr_block_idx)
        np.random.seed(42)
        sample_blocks = np.random.choice(curr_block_idx, int(
            curr_block_idx * divisor_decompression), replace=False)
        for i in sample_blocks:
            generated_file = generated_filename(dataset_name,
                                                compressor_no_flags, i, block_size_str, technique_name,
                                                total_uncompressed_size_GiB)
            # exec_cmd('tar -xf {} -I{}'.format(generated_file, chosen_compressor))
            executor.submit(decompress_one_block,
                            generated_file, chosen_compressor)

    # print("Uncompressed {} blocks".format(len(sample_blocks)))

    sample_time = float(time.time() - start_time)
    decompression_time = sample_time / divisor_decompression

    time_to_decompress_a_block = sample_time / \
                                 float(curr_block_idx * divisor_decompression)

    if keep_tar:
        # copy generated files to the output directory
        with ThreadPoolExecutor(NUM_THREAD) as executor:
            for i in range(0, curr_block_idx):
                generated_file = generated_filename(
                    dataset_name, compressor_no_flags, i, block_size_str, technique_name, total_uncompressed_size_GiB)
                # shutil.copy(generated_file, os.path.join(
                #     output_dir, os.path.basename(generated_file)))
                executor.submit(shutil.copy, generated_file, os.path.join(
                    output_dir, os.path.basename(generated_file)))

        filename_archive_map_name = 'filename_archive_map_{}_{}_{}GiB_{}.txt'.format(
            dataset_name, technique_name, str(round(total_uncompressed_size_GiB)), block_size_str)
        # print filename_archive_map on a file in the output directory separated by /n
        with open(os.path.join(output_dir, filename_archive_map_name), 'w') as f:
            for item in filename_archive_map:
                f.write(item[0] + ' ' + item[1] + os.linesep)

    os.chdir('..')
    exec_cmd('rm -rf {}'.format(path_working_dir))

    # block_size_MiB = block_size / float_1MiB
    decompression_speed = total_uncompressed_size_MiB / \
                          float(decompression_time)

    throughput_files_per_second = num_files / float(decompression_time)

    # "DATASET,NUM_FILES,TOTAL_SIZE(GiB),AVG_FILE_SIZE(KiB),MEDIAN_FILE_SIZE(KiB),TECHNIQUE,COMPRESSION_RATIO(%),ORDERING_TIME(s),COMPRESSION_TIME(s),COMPRESSION_SPEED(MiB/s),FULL_DECOMPRESSION_SPEED(MiB/s),TIME_FILE_DECOMPRESSION(ms),THROUGHPUT(files/s),COMMIT_HASH,NOTES"
    if notes == 'None':
        notes = 'block_size=' + block_size_str
    else:
        notes += '_block_size=' + block_size_str

    print('{},{},{},{},{},{}+{},{},{},{},{},{},{},{},{},{}'.format(
        dataset_name,
        len(df.index),
        str(round(total_uncompressed_size_GiB, 2)),
        str(round(avg_file_sizeKiB, 2)),
        str(round(median_file_sizeKiB, 2)),
        technique_name,
        compressor_no_flags,
        str(round((compressed_size / total_uncompressed_size * 100), 2)),  # compression ratio
        str(round(sorting_time, 2)),  # sorting time
        str(round(compression_time, 2)),  # compression time
        str(round(total_uncompressed_size_MiB /
                  float(compression_time + sorting_time), 2)),  # compression speed
        str(round(decompression_speed, 2)),  # decompression speed
        # str(round((block_size_MiB / float(decompression_speed)) * 1000., 2)), # time to decompress a block
        str(round(time_to_decompress_a_block * 1000., 2)),  # time to decompress a block
        str(round(throughput_files_per_second, 2)),
        repo.head.object.hexsha[:7],
        notes), flush=True)

    os.chdir(REPO_DIR)
