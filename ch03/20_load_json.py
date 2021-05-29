import json

def readlines_to_list(filepath: str) -> list:
    with open(filepath, 'r') as f:
        data = f.readlines()
    return data

if __name__ == '__main__':
    filepath_in = './data/jawiki-country.json'
    filepath_out = './data/jawiki-country-uk.txt'

    data_list = readlines_to_list(filepath=filepath_in)
    for data_str in data_list:
        json_dic = json.loads(data_str)
        with open(filepath_out, 'a') as f:
            if json_dic['title'] == 'イギリス':
                #print(json_dic['text'])
                f.write(json_dic['text'])
