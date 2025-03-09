import pandas as pd
import glob

dst_dir = 'merged_blobs'

# retrieve all csv files and concatenate them into dataframe
df = pd.concat([pd.read_csv(f, usecols=range(3)) for f in glob.glob('blobs-Python/*.csv')], ignore_index=True)

df.drop_duplicates('blob_hash')['blob_bytes'].sum()