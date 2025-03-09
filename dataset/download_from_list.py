#!/usr/bin/env python3
import os
import pandas as pd
import boto3
import botocore
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, wait
import subprocess
import datetime
import gzip
import shutil
import hashlib
import time
import multiprocessing
import argparse
import sys
from smart_open import open

STEP_PRINT = 100000

NUM_THREAD = 16

REPO_DIR = os.path.dirname(os.path.realpath(__file__))


def exec_cmd(cmd, to_print=False):
    if to_print:
        print('Executing -> ' + cmd)
        print('Executing -> ' + str(cmd.split()))
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    process.wait()
    output = process.communicate()
    if process.returncode != 0:
        print('[ERROR]' + output.decode('utf-8'))


chars = ['0', '1', '2', '3', '4', '5', '6',
         '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

directories = [c1+c2 for c1 in chars for c2 in chars]

# Instantiate the parser
parser = argparse.ArgumentParser(description='This script is used to download and check the integrity of the blobs '
                                 '(in the list csv-file) from the SWH dataset.')

# get files lists
parser.add_argument('csv_file_list', metavar="csv-file", nargs='+', default=[os.path.join(REPO_DIR, 'examples/random_small.csv')],
                    help='List of files (in csv format) to compress')

parser.add_argument('--output', default='/home/boffa/Experiments/acubeLab/PPC_utils4BigData/tmp/BLOBS',
                    help='Dir where the uncompressed blobs are stored')

parser.add_argument('-T', '--num-thread', default=4, type=int,
                    help='Number of thread used to download decompress check count blobs\n')

parser.add_argument('--max-download', default=str(2**40),
                    help='Maximum amount of bytes to download')


args = parser.parse_args()

if len(args.csv_file_list) == 0:
    parser.print_help()
    exit(1)


# Get file's Last modification time stamp only in terms of seconds since epoch
modTimesinceEpoc = os.path.getmtime(__file__)
# Convert seconds since epoch to readable timestamp
modificationTime = time.strftime(
    '%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
print('\n# {} Last Modified Time : '.format(
    os.path.basename(__file__)), modificationTime)
print('#Starting time : ', time.strftime(
    '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

OUTPUT_DIR = args.output

MAX_DOWNLOAD = int(args.max_download)

NUM_THREAD = int(args.num_thread)

print(
    f"#Machine  {os.uname()[1]} blobs downloaded in {OUTPUT_DIR} PID {os.getpid()}", flush=True)

print('NUM THREADS = ' + str(NUM_THREAD))

os.makedirs(OUTPUT_DIR, exist_ok=True)

for d in directories:
    dir_path = os.path.join(OUTPUT_DIR, d)
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(f"Fatal: Cannot create output directory {dir_path}\n", e)
        sys.exit(1)

for dataset in args.csv_file_list:
    dataset_name = os.path.basename(dataset)

    df = pd.read_csv(dataset,
                     dtype={'swhid': 'string',
                            'file_id': 'string',
                            'length': 'Int64',
                            'local_path': 'string',
                            'filename': 'string',
                            'filepath': 'string'},
                     on_bad_lines='skip',
                     engine='python',
                     encoding_errors='ignore', index_col=0)

    print('Dataset {} loaded. Stats BEFORE dropna'.format(
        dataset_name), flush=True)

    print('Num files ' + str(len(df.index)))
    print('Num bytes ' + str(df['length'].sum()))

    num_blobs = len(df.index)

    size_in_bytes = int(df['length'].sum())
    print(dataset_name + ' has ' + str(size_in_bytes) +
          ' bytes. In GiB ' + str(float(size_in_bytes) / float(2**30)))

    downloaded_objects = 0
    downloaded_objects_size = 0

    present_objects = 0
    present_objects_size = 0

    # it's submitted_jobs more than downloaded_objects
    submitted_jobs = 0

    s3 = boto3.client('s3', config=Config(
        signature_version=botocore.UNSIGNED, max_pool_connections=50))

    STEP_PRINT = min(STEP_PRINT, int(num_blobs/100))

    # with ProcessPoolExecutor(NUM_THREAD) as executor:
    with ThreadPoolExecutor(NUM_THREAD) as executor:
        def download_decompress_object_boto3(file_sha1):
            # Downloads and decompress an object from S3 to local
            # print('Downloading (with boto3) {}'.format(filepath))
            try:
                s3_url = f"s3://softwareheritage/content/{file_sha1}"
                file_on_filesystem = os.path.join(
                    OUTPUT_DIR, file_sha1[:2], file_sha1)

                with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
                    with open(file_on_filesystem, "wb") as fout:
                        fout.write(fin.read())

                global downloaded_objects
                global downloaded_objects_size
                downloaded_objects_size += int(
                    os.path.getsize(file_on_filesystem))
                downloaded_objects += 1
            except Exception as e:
                print(
                    str(e) + ' Error while dowloading (with boto3) ' + file_on_filesystem)

        prev_time = time.time()
        prev_blobs = 0
        prev_MiB = 0

        df.reset_index(drop=True, inplace=True)

        for i, row in df.iterrows():
            file_sha1 = row['file_id']
            # file_sha1[:2] for the two level storing scheme à la Git
            if i % STEP_PRINT == 0:
                # if i % 100 == 0:
                curr_time = time.time()
                curr_blobs = downloaded_objects
                curr_MiB = downloaded_objects_size / float(2 ** 20)

                print('{} (download if not exist or size 0) {} / {}. (Queue size {}) {}% (Downloaded compressed objects {}, {} bytes) {}). blobs per second {}. MiB/s {}'.format(
                    file_sha1,
                    str(i),
                    str(num_blobs),
                    executor._work_queue.qsize() if hasattr(
                        executor, '_work_queue') else 0,
                    str((float(i) / float(num_blobs)) * 100.)[:5],
                    downloaded_objects,
                    downloaded_objects_size,
                    datetime.datetime.now(),
                    (curr_blobs - prev_blobs) / (curr_time - prev_time),
                    (curr_MiB - prev_MiB) / (curr_time - prev_time)),
                    flush=True)

                prev_time = time.time()
                prev_blobs = downloaded_objects
                prev_MiB = downloaded_objects_size / float(2 ** 20)

            file_path = os.path.join(OUTPUT_DIR, file_sha1[:2], file_sha1)
            if not (os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
                submitted_jobs += 1
                if NUM_THREAD == 1:
                    download_decompress_object_boto3(file_sha1)
                else:
                    executor.submit(download_decompress_object_boto3,
                                    file_sha1)
            else:
                present_objects += 1
                present_objects_size += os.path.getsize(file_path)

    print(f"Done download {dataset}", flush=True)
    print(f"Downloaded {downloaded_objects} objects, {downloaded_objects_size} bytes", flush=True)
    print(f"Preset {present_objects} objects, {present_objects_size} bytes", flush=True)

