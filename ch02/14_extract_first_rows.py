import filecmp
import pandas as pd
import sys
import subprocess

def extract_first_rows(filepath: str, n_row: int) -> pd.DataFrame:
    df_in = pd.read_table(filepath, header=None)
    df_out = df_in.iloc[:n_row, :]
    return df_out

if __name__ == '__main__':
    n_row = sys.argv[1]
    filepath_in = './data/popular-names.txt'
    filepath_out_py = './data/popular-names-14-py.txt'
    filepath_out_unix = './data/popular-names-14-unix.txt'

    df = extract_first_rows(filepath=filepath_in, n_row=int(n_row))
    df.to_csv(filepath_out_py, sep='\t', index=False, header=False)

    # head -n 10 ./data/popular-names.txt > ./data/popular-names-14-unix.txt
    with open (filepath_out_unix, 'w') as f:
        subprocess.run(["head", "-n", n_row, filepath_in], stdout=f)
    assert filecmp.cmp(filepath_out_py, filepath_out_unix)
    print('All test cases passed.')