import filecmp
import pandas as pd
import sys
import subprocess

def split_into_files(filepath_in: str, n_split: int) -> None:
    df_in = pd.read_table(filepath_in, header=None)
    n_row = len(df_in)
    n_row_per_file = (n_row + n_split - 1) // n_split
    for i in range(n_split):
        df_out = df_in.iloc[i*n_row_per_file:min((i+1)*n_row_per_file, n_row), :]
        filepath_out = filepath_in.replace('.txt', '-py-part' + str(i).zfill(2) + '.txt')
        df_out.to_csv(filepath_out, header=None, index=False, sep='\t')

if __name__ == '__main__':
    n_split = sys.argv[1]
    filepath_in = './data/popular-names.txt'

    split_into_files(filepath_in=filepath_in, n_split=int(n_split))

    # gsplit -d -l $(((`wc -l < "./data/popular-names.txt"` + 7 - 1) / 7)) ./data/popular-names.txt ./data/popular-names-unix-part --additional-suffix=.txt

    for i in range(int(n_split)):
        filepath_out_py = filepath_in.replace('.txt', '-py-part' + str(i).zfill(2) + '.txt')
        filepath_out_unix = filepath_in.replace('.txt', '-unix-part' + str(i).zfill(2) + '.txt')
        assert filecmp.cmp(filepath_out_py, filepath_out_unix)

    print('All test cases passed.')