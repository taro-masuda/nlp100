import subprocess
import filecmp
import pandas as pd

def extract_col(idx_col: int, filepath_in: str) -> str:
    df = pd.read_table(filepath_in, header=None)
    df_1col = df.iloc[:, idx_col]
    filepath_out = filepath_in.replace('popular-names.txt', 'col' + str(idx_col+1) + '.txt')
    df_1col.to_csv(filepath_out, index=False, header=None)
    return filepath_out

def split_columns(filepath_in: str, idx_col: int) -> str:
    return extract_col(idx_col=idx_col, filepath_in=filepath_in)

if __name__ == '__main__':
    filepath_in = './data/popular-names.txt'

    filepath_out_py = [split_columns(filepath_in=filepath_in, idx_col=0)]
    filepath_out_py.append(split_columns(filepath_in=filepath_in, idx_col=1))

    # cut -f 1 ./data/popular-names.txt > ./data/popular-names-12-unix-0.txt
    # cut -f 2 ./data/popular-names.txt > ./data/popular-names-12-unix-1.txt
    for i in range(len(filepath_out_py)):
        filepath_out_unix = filepath_in.replace('.txt', '-12-unix-' + str(i) + '.txt')
        assert filecmp.cmp(filepath_out_py[i], filepath_out_unix)
    print('All test cases passed.')