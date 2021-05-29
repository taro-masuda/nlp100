import re

def extract_categories(filepath: str) -> None:
    reg = re.compile("\[\[Category:.*\]\]")
    with open(filepath, 'r') as f:
        str_in = f.read()
    categories = reg.findall(str_in)
    for category in categories:
        print(category)

if __name__ == '__main__':
    filepath_in = './data/jawiki-country-uk.txt'
    extract_categories(filepath=filepath_in)