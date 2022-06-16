import re
import numpy as np
import random
import os
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import math

java_keywords = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case',
                 'catch', 'char', 'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
                 'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 'import',
                 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private',
                 'protected', 'public', 'return', 'strictfp', 'short', 'static', 'super', 'switch', 'synchronized',
                 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while']


def camel_case_split(str):
    RE_WORDS = re.compile(r'''
    [A-Z]+(?=[A-Z][a-z]) |
    [A-Z]?[a-z]+ |
    [A-Z]+ |
    \d+ |
    [^\u4e00-\u9fa5^a-z^A-Z^0-9]+
    ''', re.VERBOSE)
    return RE_WORDS.findall(str)


class StatementAnalyse():
    def __init__(self, output_dir="./output"):
        self.output_dir = output_dir
        self.tokenMap = {}
        self.globalTokenAttention = {}

    def statement_reader(self, statement_index, layer_num):
        statements = []
        with open(self.output_dir + "/" + str(statement_index) + '/layer_' + str(layer_num), 'r') as f:
            item = "blank_item"
            while True:
                item = f.readline().rstrip('\n').replace('tensor(', '').replace('device=\'cuda:1\'),', '')
                if not item:
                    break
                statements.append(eval(item))
        self.statements = statements
        return statements

    def read_global_token_attention(self, token, filename='./token_attention'):
        if len(self.globalTokenAttention) == 0:
            with open(filename, 'r') as f:
                item = 'blank_item'
                while True:
                    item = f.readline()
                    key = item.split('  ||  ')[0]
                    value = float(item.split('  ||  ')[1])
                    if not item:
                        break
                    self.globalTokenAttention[key] = value
        return self.globalTokenAttention[token]

    def token_frequence_attention_map(self, tokens, statement_index, max_layer_num=12):
        statement_attentions = []
        statement_attentions = [self.statement_reader(statement_index, i)
                                for i in range(max_layer_num-1, max_layer_num)]
        average_attention = np.mean(np.array(statement_attentions), axis=0)
        for index, statement in enumerate(average_attention):
            need_calculate = True

            i = 0

            start = 0
            end = 0
            while i < len(statement):
                if statement[i] == 0.0:
                    if not need_calculate:
                        break
                    i = i + 1
                    continue
                else:
                    need_calculate = False

                    start = i
                    while i < len(statement):
                        if statement[i] == 0.0:
                            end = i
                            break
                        if tokens[i+1].startswith('Ġ'):
                            end = i + 1
                            break
                        else:
                            i = i + 1

                    token = ''.join(tokens[start:end])

                    camelCaseWord = []
                    # separate camel case
                    if token.startswith('Ġ'):
                        if len(token) > 1:
                            camelCaseWord = camel_case_split(token[1:])
                    else:
                        camelCaseWord = camel_case_split(token)

                    for token in camelCaseWord:
                        if token in self.tokenMap.keys():
                            frequence = self.tokenMap[token]['frequence']
                            attention = self.tokenMap[token]['attention']
                            self.tokenMap[token]['attention'] = \
                                (attention * frequence +
                                 sum(average_attention[index][start:end])/(end-start)) / (frequence + 1)
                            self.tokenMap[token]['frequence'] = frequence + 1
                        else:
                            self.tokenMap[token] = {'frequence': 1, 'attention': sum(
                                average_attention[index][start:end])/(end-start)}
                    start = end
                    i = i + 1

    def get_statement_attention(self, tokens, statement_index, in_statement_modulu=0.7, max_layer_num=12,
                                single_layer=False, layer_num=0):
        statement_attentions = []
        if not single_layer:
            statement_attentions = [self.statement_reader(statement_index, i) for i in range(0, max_layer_num)]
        else:
            statement_attentions = [self.statement_reader(statement_index, layer_num)]
        average_attention = np.mean(np.array(statement_attentions), axis=0) * in_statement_modulu
        statement_attention = []
        for index, statement in enumerate(average_attention):
            need_calculate = True

            # calculate the average attention of the whole statement
            attention_sum = 0.0
            token_num = 0

            token_list = []

            for i in range(0, len(statement)):
                if statement[i] == 0.0:
                    if not need_calculate:
                        break
                    continue
                else:
                    # calculate the tokens' attention with the ratio of the whole dictionary and the attention in the
                    # statement
                    token_attention = self.global_token_attention_map[tokens[i]] * (1-in_statement_modulu)
                    average_attention[index][i] += token_attention

                    token_list.append(tokens[i])

                    attention_sum += average_attention[index][i]
                    token_num += 1

                    need_calculate = False
            statement_attention.append({"token": token_list, "attention": (attention_sum / token_num)})
        return average_attention, statement_attention

    def output_statement(self, tokens, statement_index, output_file_dir, in_statement_modulu=0.7, max_layer_num=12):
        _, statement_attention = self.get_statement_attention(tokens,
                                                              statement_index,
                                                              in_statement_modulu, max_layer_num)
        with open(output_file_dir, 'a') as f:
            for statement in statement_attention:
                token = statement['token']
                attention = statement['attention']
                f.write(str(token) + '\n')
                f.write(str(attention) + '\n')


def token_reader(statement_index, output_dir="./output"):
    tokens = []
    with open(output_dir + "/" + str(statement_index) + '/tokens', 'r') as f:
        item = "blank_item"
        while True:
            item = f.readline().rstrip('\n')
            if not item:
                break
            tokens.append(item)
    return tokens


def global_token_attention_reader(output_dir="./weights", filename="latest_output"):
    print("start loading attentions...")
    function_num = len(os.listdir('./output'))
    sa = StatementAnalyse()
    for filename in range(1, function_num + 1):
        token = token_reader(int(filename))
        sa.token_frequence_attention_map(token, int(filename))
    print("finish loading attentions...")
    outputMap = {}
    for key in sa.tokenMap.keys():
        if 'Ġ' not in key:
            continue
        attention = sa.tokenMap[key]['attention']
        frequence = sa.tokenMap[key]['frequence']
        if key in outputMap.keys():
            current_attention = outputMap[key]['attention']
            current_frequence = outputMap[key]['frequence']
            outputMap[key]['attention'] = \
                (current_frequence * current_attention + attention) / (current_frequence + frequence)
            outputMap[key]['frequence'] = current_frequence + frequence
        else:
            outputMap[key] = {'attention': attention, 'frequence': frequence}
    return outputMap


def my_tf_color_func(dictionary):
    def my_tf_color_func_inner(word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(0, 0%%, %d%%)" % (random.randint(20, 60))
    return my_tf_color_func_inner


def overall_analyse(output_dir="./output"):
    g = os.walk(output_dir)
    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            print(dir_name)


def get_cloud_item(content, key):
    if key not in content.keys():
        return 0
    return int(math.log(content[key]['attention'] * 100000, 2))


def get_cloud(content):
    data = ""
    with open('./word.txt', 'w') as f:
        for key in content.keys():
            if content[key]['frequence'] > 100:
                data += (key[1:] + " \n") * get_cloud_item(content, key)
        f.flush()
        f.write(data)
    f = open(u'./word.txt', 'r').read()
    wordcloud = WordCloud(background_color="white", width=1200, height=960, margin=1,
                          collocations=False, color_func=my_tf_color_func(content)).generate(f)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    wordcloud.to_file('test.png')
    f.close()


def get_keyword_cloud(content):
    data = ""
    with open('./word.txt', 'w') as f:
        data += "for \n" * get_cloud_item(content, "Ġfor")
        data += "if \n" * get_cloud_item(content, "Ġif")
        data += "while \n" * get_cloud_item(content, "Ġwhile")
        data += "public \n" * get_cloud_item(content, "Ġpublic")
        data += "private \n" * get_cloud_item(content, "Ġprivate")
        data += "abstract \n" * get_cloud_item(content, "Ġabstract")
        data += "extends \n" * get_cloud_item(content, "Ġextends")
        data += "protected \n" * get_cloud_item(content, "Ġprotected")
        data += "int \n" * get_cloud_item(content, "Ġint")
        data += "float \n" * get_cloud_item(content, "Ġfloat")
        data += "boolean \n" * get_cloud_item(content, "Ġboolean")
        data += "break \n" * get_cloud_item(content, "Ġbreak")
        data += "byte \n" * get_cloud_item(content, "Ġbyte")
        data += "catch \n" * get_cloud_item(content, "Ġcatch")
        data += "else \n" * get_cloud_item(content, "Ġelse")
        data += "final \n" * get_cloud_item(content, "Ġfinal")
        data += "finally \n" * get_cloud_item(content, "Ġfinally")
        data += "new \n" * get_cloud_item(content, "Ġnew")
        data += "return \n" * get_cloud_item(content, "Ġreturn")
        data += "try \n" * get_cloud_item(content, "Ġtry")
        data += "void \n" * get_cloud_item(content, "Ġvoid")
        data += "native \n" * get_cloud_item(content, "Ġnative")
        data += "static \n" * get_cloud_item(content, "Ġstatic")
        data += "transient \n" * get_cloud_item(content, "Ġtransient")
        data += "volatile \n" * get_cloud_item(content, "Ġvolatile")
        data += "do \n" * get_cloud_item(content, "Ġdo")
        data += "instanceof \n" * get_cloud_item(content, "Ġinstance")
        data += "switch \n" * get_cloud_item(content, "Ġswitch")
        data += "case \n" * get_cloud_item(content, "Ġcase")
        data += "default \n" * get_cloud_item(content, "Ġdefault")
        data += "throw \n" * get_cloud_item(content, "Ġthrow")
        data += "super \n" * get_cloud_item(content, "Ġsuper")
        data += "this \n" * get_cloud_item(content, "Ġthis")
        data += "null \n" * get_cloud_item(content, "Ġnull")
        data += "true \n" * get_cloud_item(content, "Ġtrue")
        data += "false \n" * get_cloud_item(content, "Ġfalse")
        data += "short \n" * get_cloud_item(content, "Ġshort")
        data += "long \n" * get_cloud_item(content, "Ġlong")
        data += "double \n" * get_cloud_item(content, "Ġdouble")
        data += "char \n" * get_cloud_item(content, "Ġchar")
        f.flush()
        f.write(data)
    f = open(u'./word.txt', 'r').read()
    wordcloud = WordCloud(background_color="white", width=1000, height=860, margin=1, collocations=False).generate(f)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    wordcloud.to_file('test.png')
    f.close()


def get_item(content, key):
    return content["Ġ" + key]


# attention_map = global_token_attention_reader()

# get_cloud(attention_map)


def output_token_frequence_and_attention(output_image=False):
    print("start loading attentions...")
    function_num = len(os.listdir('./output'))
    sa = StatementAnalyse()
    x = []
    y = []
    annotation = []
    for filename in range(1, function_num + 1):
        token = token_reader(int(filename))
        sa.token_frequence_attention_map(token, int(filename))
    print("finish loading attentions...")
    outputMap = {}
    for key in sa.tokenMap.keys():
        if key in java_keywords:
            attention = sa.tokenMap[key]['attention']
            frequence = sa.tokenMap[key]['frequence']
            if frequence in outputMap.keys():
                current_attention = outputMap[frequence]['attention']
                current_repeat = outputMap[frequence]['repeat']
                outputMap[frequence]['attention'] = \
                    (current_repeat * current_attention + attention) / (current_repeat + 1)
                outputMap[frequence]['repeat'] = current_repeat + 1
            else:
                outputMap[frequence] = {'attention': attention, 'repeat': 1, 'name': key}
    for key in outputMap.keys():
        attention = outputMap[key]['attention']
        if key >= 1:
            x.append(key * 150)
            y.append(attention)
            annotation.append(outputMap[key]['name'])

    if output_image:
        print("drawing...")
        # fig, ax = plt.subplots()
        # ax.scatter(x, y)
        # print(annotation)
        # for i in range(len(x)):
        # plt.annotate(annotation[i], xy=(x[i], y[i]), xytext=(x[i]+1, y[i]))
        # plt.show()

        result = sorted(outputMap.items(), key=lambda item: item[1]['attention'], reverse=True)
        x = [a[1]['name'] for a in result]
        y = [a[1]['attention']-0.002 for a in result]
        with open('./java-keyword-token', 'w') as f:
            for i in range(len(x)):
                f.write(str(x[i]) + '\n')
            for i in range(len(x)):
                f.write(str(y[i]) + '\n')
#         plt.bar(x, y, width=0.5, align='center')
        # plt.xticks(rotation=45)
#         plt.show()


def generate_token_attention_file(output_file_dir='./token_attentions'):
    print("start loading attentions...")
    function_num = len(os.listdir('./output'))
    sa = StatementAnalyse()
    for filename in range(1, function_num + 1):
        token = token_reader(int(filename))
        sa.token_frequence_attention_map(token, int(filename))
    attentions = sorted(sa.tokenMap.items(), key=lambda item: item[1]['attention'], reverse=False)
    with open(output_file_dir, 'w') as f:
        for token in attentions:
            f.write(str(token[0]) + '  ||  ' + str(token[1]['attention']) + '\n')


def generate_output_statement_attention():
    function_num = len(os.listdir('./output'))
    sa = StatementAnalyse()
    attention_list = []

    print("loading function files...")
    for filename in range(1, function_num + 1):
        token = token_reader(int(filename))
        _, statement_attention = sa.get_statement_attention(token, int(filename))
        attention_list = attention_list + statement_attention
    print("sorting attention")
    sorted_attention_list = sorted(attention_list, key=lambda statement: statement['attention'])
    print("output statement")
    with open('./output_statement/statement_attention', 'a') as f:
        for statement in sorted_attention_list:
            token = ' '.join(statement['token']).replace('Ġ', '')
            if token == "}":
                continue
            attention = statement['attention']
            f.write(str(attention) + '  ||  ' + str(token) + '\n')

    for layer_num in range(0, 12):
        attention_list = []
        print("start process single attention layer")
        print("loading function files...")
        for filename in range(1, function_num + 1):
            token = token_reader(int(filename))
            _, statement_attention = sa.get_statement_attention(
                token, int(filename), single_layer=True, layer_num=int(layer_num))
            attention_list = attention_list + statement_attention
            sa.output_statement(token, int(filename), './output_statement/' + str(filename))
        print("sorting attention")
        sorted_attention_list = sorted(attention_list, key=lambda statement: statement['attention'])
        print("output statement")
        with open('./output_statement/statement_attention_layer_' + str(layer_num), 'a') as f:
            for statement in sorted_attention_list:
                token = ' '.join(statement['token']).replace('Ġ', '')
                if token == "}":
                    continue
                attention = statement['attention']
                f.write(str(attention) + '  ||  ' + str(token) + '\n')


if __name__ == '__main__':
    generate_token_attention_file()
    # output_token_frequence_and_attention(output_image=True)
    pass
