import subprocess
import filecmp

def replace_tabs_into_spaces(in_str: str) -> str:
    out_string = in_str.replace('\t', ' ')
    return out_string

if __name__ == '__main__':
    filepath_in = './data/popular-names.txt'
    filepath_out_py = filepath_in.replace('popular-names.txt', 'popular-names-py.txt')
    filepath_out_unix = filepath_in.replace('popular-names.txt', 'popular-names-unix.txt')

    with open(filepath_in, 'r') as f:
        in_str = f.read()
    out_str_py = replace_tabs_into_spaces(in_str=in_str)
    with open(filepath_out_py, 'w') as f:
        f.write(out_str_py)
    
    # sed -e "s/      / /g" ./data/popular-names.txt > ./data/popular-names-unix.txt
    assert filecmp.cmp(filepath_out_py, filepath_out_unix)
    print('All test cases passed.')