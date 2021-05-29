import re

def extract_category_names(filepath: str) -> None:
    reg = re.compile("\[\[Category:.*\]\]")
    with open(filepath, 'r') as f:
        str_in = f.read()
    categories = reg.findall(str_in)
    for category in categories:
        cate_print = re.sub('\[\[Category:', '', category)
        cate_print = re.sub('\]\]', '', cate_print)
        print(cate_print)

if __name__ == '__main__':
    filepath_in = './data/jawiki-country-uk.txt'
    extract_category_names(filepath=filepath_in)