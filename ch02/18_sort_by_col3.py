import filecmp
import pandas as pd

def sort_by_col3(filepath_in: str) -> pd.DataFrame:
    df_in = pd.read_table(filepath_in, header=None)
    df_sort = df_in.sort_values([2, 0], ascending=[False, True])

    return df_sort

if __name__ == '__main__':
    filepath_in = './data/popular-names.txt'
    filepath_out_py = './data/popular-names-sort-by-col3-py.txt'
    filepath_out_unix = './data/popular-names-sort-by-col3-unix.txt'

    df = sort_by_col3(filepath_in=filepath_in)
    df.to_csv(filepath_out_py, index=None, header=None, sep='\t')

    # sort -b -k 3,3rn -k 1,1 ./data/popular-names.txt > ./data/popular-names-sort-by-col3-unix.txt

    assert filecmp.cmp(filepath_out_py, filepath_out_unix)

    print('All test cases passed.')