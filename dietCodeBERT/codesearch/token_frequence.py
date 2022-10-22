import os
import json
from transformers import RobertaTokenizer


def remove_annotation(statements):
    inAnnotation = False
    for index, statement in enumerate(statements):
        if inAnnotation:
            if statement.endswith('*/'):
                inAnnotation = False
                statements[index] = ''
            elif '*/' in statement:
                inAnnotation = False
                statements[index] = statements[index][statements[index].find('*/') + 1:]
            else:
                statements[index] = ''
        elif not inAnnotation:
            if statement.startswith('//'):
                statements[index] = ''
            elif statement.startswith('/*'):
                statements[index] = ''
                inAnnotation = True
            elif '/*' in statement:
                statements[index] = statements[index][:statements[index].find('/*') - 1]
                inAnnotation = True
            elif '// ' in statement:
                statements[index] = statements[index][:statements[index].find('//') - 1]
    return [s for s in statements if s != '']


def generate_freq_token():
    data_dir = '../data/java/final/jsonl/train'

    g = os.walk(data_dir)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    token_map = {}

    for path, dir_list, file_list in g:
        for file_name in file_list:
            file_dir = os.path.join(path, file_name)
            with open(file_dir, 'r') as f:
                print('Loading file ' + str(file_dir))
                data = f.readlines()
                for d in data:
                    code = json.loads(d)['code']
                    statements = [x.strip() for x in code.replace('\t', ' ').split('\n')]
                    statements = remove_annotation(statements)
                    code = ' '.join([s for s in statements if s != ''])
                    tokens = tokenizer.tokenize(code)
                    for token in tokens:
                        if token[0] == 'Ġ':
                            token = token[1:]
                        if token in token_map:
                            token_map[token] += 1
                        else:
                            token_map[token] = 1
            with open('./token_frequence', 'w') as f:
                print("Writing tokens...")
                for k, v in token_map.items():
                    f.write(str(k) + '  ||  ' + str(v) + '\n')


def read_token_freq(filename='./token_frequence'):
    token_map = {}
    with open(filename, 'r') as f:
        line = f.readline()
        while line is not None and line != '':
            token = line.split('  ||  ')[0]
            freq = line.split('  ||  ')[1]
            token_map[token] = int(freq)
            line = f.readline()
    return token_map


def prune_token(tokens, token_map, rate = 0.5):
    freq = []
    for token in tokens:
        if token not in token_map.keys():
            freq.append(0)
        else:
            freq.append(token_map[token])
    import copy
    freq_ = copy.deepcopy(freq)
    freq_ = sorted(freq_)
    threshold = freq_[int(len(freq_) * rate)]
    result = []
    for index, token in enumerate(tokens):
        if freq[index] >= threshold:
            result.append(token)
    return result


if __name__ == '__main__':
    pass
    # generate_freq_token()
    # token_map = read_token_freq()
