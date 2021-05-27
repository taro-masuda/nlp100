import pandas as pd
import subprocess

def count_row(filepath: str) -> int:
    df = pd.read_table(filepath, header=None)
    return len(df)
def count_row_unix(filepath: str) -> bytes:
    res = subprocess.run(['wc', '-l', filepath], stdout=subprocess.PIPE)
    return res.stdout

if __name__ == '__main__':
    print('Pandas result: ', count_row('./data/popular-names.txt'))
    print('Unix result: ', count_row_unix('./data/popular-names.txt'))