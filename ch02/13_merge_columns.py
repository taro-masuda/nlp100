import filecmp
import pandas as pd

def merge_col(filepath1: str, filepath2: str) -> pd.DataFrame:
    df1 = pd.read_table(filepath1, header=None)
    df2 = pd.read_table(filepath2, header=None)
    df_out = pd.concat([df1, df2], axis=1)
    return df_out

if __name__ == '__main__':
    filepath1 = './data/col1.txt'
    filepath2 = './data/col2.txt'
    filepath_out_py = './data/popular-names-13-py.txt'
    filepath_out_unix = './data/popular-names-13-unix.txt'
    df = merge_col(filepath1=filepath1, filepath2=filepath2)
    df.to_csv(filepath_out_py, sep='\t', index=False, header=False)

    # paste ./data/col1.txt ./data/col2.txt > ./data/popular-names-13-unix.txt
    assert filecmp.cmp(filepath_out_py, filepath_out_unix)
    print('All test cases passed.')