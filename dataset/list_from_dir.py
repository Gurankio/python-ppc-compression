#!/usr/bin/env python3
import argparse
import os
import pandas as pd

Description = ''' 
This script lists all files in a directory and its subdirectories 
and saves the list to a csv file compatible with bench_PPC.py
'''

parser = argparse.ArgumentParser(
    description=Description, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('root_directory', metavar="root_directory", nargs='+',
                    help='')

args = parser.parse_args()

if __name__ == "__main__":

    root_full_path = args.root_directory[0]
    root_dir = os.path.basename(root_full_path)
    # print(root_full_path)

    df = pd.DataFrame(
        columns=['swh_id', 'file_id', 'length', 'filename', 'filepath', 'local_path'])

    # recursivelly list all files in the directory
    for directory, dirs, files in os.walk(root_full_path):
        for file in files:
            file_full_path = os.path.join(directory, file)
            local_path = directory.replace(root_full_path + '/', "")
            file_length = os.path.getsize(file_full_path)
            new_row = {'swh_id': '0', 'file_id': os.path.basename(file), 'length': file_length,
                       'filename': os.path.basename(file), 'filepath': os.path.join(local_path, file), 'local_path': local_path}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(f'{root_dir}_list_of_files.csv', index=False)
