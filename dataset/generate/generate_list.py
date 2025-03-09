#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import os


def main():
    df = pd.DataFrame(columns=['blob_hash', 'blob_bytes', 'blob_path', 'local_path'])

    for file in (Path(__file__) / 'sources').glob('*.csv'):
        # TODO: quote char?
        df_tmp = pd.read_csv(file, on_bad_lines='warn', engine='python', encoding_errors='ignore')
        df_tmp['local_path'] = file.stem
        df = pd.concat([df, df_tmp])

    print(df.head())
    print(df.drop_duplicates('blob_hash')['blob_bytes'].sum())

    # Create a new dataframe
    df_new = pd.DataFrame(columns=['swh_id', 'file_id', 'length', 'filename', 'filepath', 'local_path'])
    df_new['file_id'] = df['blob_hash']
    df_new['length'] = df['blob_bytes']
    df_new['filepath'] = df['blob_path']
    df_new['filename'] = df['blob_path'].apply(os.path.basename)
    df_new['local_path'] = df['local_path']
    # `swh_id` can be easily computed from file_content but here is necessary
    # Dummy value. Blobs from GitHub do not have swh_id by default
    df_new['swh_id'] = '0'

    df_new.reset_index(drop=True, inplace=True)
    print(df_new.head())

    print(f"Before drop duplicates: {df_new['length'].sum()} in {round(df_new['length'].sum() / (2 ** 30), 2)} GiB)")

    # # Create an empty DataFrame to store duplicates
    # duplicates_df = pd.DataFrame(columns=df_new.columns)

    # # Find duplicates based on 'file_id'
    # duplicate_mask = df_new.duplicated(subset='file_id', keep=False)

    # # Store duplicates in the new DataFrame
    # duplicates_df = df_new[duplicate_mask].copy()

    # # Function to find differences and store them in a new column
    # def find_differences(row):
    #     first_occurrence = df_new[df_new['file_id'] == row['file_id']].iloc[0]
    #     differences = {col: (first_occurrence[col], row[col]) for col in df_new.columns if first_occurrence[col] != row[col]}
    #     return differences

    # # Apply the function to find differences for each row of duplicates
    # duplicates_df['differences'] = duplicates_df.apply(find_differences, axis=1)

    df_new.drop_duplicates('file_id', inplace=True)
    print(f"Alfter drop duplicates: {df_new['length'].sum()} in {round(df_new['length'].sum() / (2 ** 30), 2)} GiB)")

    df_new.to_csv('50GiB_github.csv', index=False)

    # duplicates_df.to_csv('DUPLICATE_IN_50GiB_github.csv', index=False)


if __name__ == '__main__':
    main()
