import re

def extract_file_references(filepath: str) -> None:
    reg = re.compile("\[\[ファイル:.*\|?")
    with open(filepath, 'r') as f:
        str_in = f.read()
    filenames = reg.findall(str_in)
    for filename in filenames:
        filename_print = re.sub('\[\[ファイル:', '', filename)
        filename_print = re.sub('\|.*\]', '', filename_print)
        filename_print = re.sub('}', '', filename_print)
        filename_print = re.sub(']', '', filename_print)
        filename_print = re.sub('</.*><.*/>', '', filename_print)

        print(filename_print)

if __name__ == '__main__':
    filepath_in = './data/jawiki-country-uk.txt'
    extract_file_references(filepath=filepath_in)