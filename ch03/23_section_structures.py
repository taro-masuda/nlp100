import re

def section_structures(filepath: str) -> None:
    reg = re.compile("={2,}.*={2,}")
    with open(filepath, 'r') as f:
        str_in = f.read()
    categories = reg.findall(str_in)
    for category in categories:
        reg_eq = re.compile("={2,}")
        equals = reg_eq.findall(category)
        num_eq = len(equals[0])
        cate_print = re.sub('={2,}', '', category)
        print(cate_print, num_eq-1)

if __name__ == '__main__':
    filepath_in = './data/jawiki-country-uk.txt'
    section_structures(filepath=filepath_in)