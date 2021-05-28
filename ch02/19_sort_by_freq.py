import filecmp
import pandas as pd
import sys
import subprocess

def sort_by_freq(filepath_in: str) -> pd.DataFrame:
    df_in = pd.read_table(filepath_in, header=None)
    df_name = df_in.iloc[:, 0]
    df_freq = df_name.value_counts().to_frame()
    
    df_freq['index'] = df_freq.index
    df_freq = df_freq.sort_values([0, 'index'], ascending=[False, True])

    return df_freq   

if __name__ == '__main__':
    filepath_in = './data/popular-names.txt'
    filepath_out_py = './data/popular-names-sort-by-freq-py.txt'
    filepath_out_unix = './data/popular-names-sort-by-freq-unix.txt'

    df = sort_by_freq(filepath_in=filepath_in)
    df.to_csv(filepath_out_py, index=None, header=None, sep=' ')

    # gcut -f 1 ./data/popular-names.txt | sort | uniq -c | sort -b -k 1,1rn -k 2,2 | sed 's/^[ \t]*//' > ./data/popular-names-sort-by-freq-unix.txt 

    assert filecmp.cmp(filepath_out_py, filepath_out_unix)

    print('All test cases passed.')