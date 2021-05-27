import filecmp
import pandas as pd
import sys
import subprocess

def distinct_strings(filepath_in: str, filepath_out: str) -> None:
    df_in = pd.read_table(filepath_in, header=None)
    df_name = df_in.iloc[:,0]
    df_name_distinct = df_name.drop_duplicates()
    df_name_distinct = df_name_distinct.sort_values()
    df_name_distinct.to_csv(filepath_out, header=None, index=None)

if __name__ == '__main__':
    filepath_in = './data/popular-names.txt'
    filepath_out_py = './data/popular-names-uniq-py.txt'
    filepath_out_unix = './data/popular-names-uniq-unix.txt'

    distinct_strings(filepath_in=filepath_in, filepath_out=filepath_out_py)

    # cut -f 1 ./data/popular-names.txt | sort | uniq > ./data/popular-names-uniq-unix.txt

    assert filecmp.cmp(filepath_out_py, filepath_out_unix)

    print('All test cases passed.')