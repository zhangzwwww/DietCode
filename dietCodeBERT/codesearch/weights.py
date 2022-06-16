class WeightOutputer():
    def __init__(self):
        self.index = 2
        self.tokenMap = {}
        self.outputFileDir = ''

    def set_output_file_dir(self, fileDir):
        self.outputFileDir = fileDir

    def init_tokenMap(self, tokens):
        for token in tokens:
            if token not in self.tokenMap.keys():
                self.tokenMap[token] = {'frequence': 0, 'attention': 0}

    def update_tokenMap(self, attentions, tokens):
        self.init_tokenMap(tokens)
        for i in range(0, len(attentions)):
            frequence = self.tokenMap[tokens[i]]['frequence']
            attention = self.tokenMap[tokens[i]]['attention']
            self.tokenMap[tokens[i]]['attention'] = (attention * frequence + attentions[i].item()) / (frequence + 1)
            self.tokenMap[tokens[i]]['frequence'] += 1

    def generate_attention_map(self):
        self.set_output_file_dir('./weights')
        f = open(self.outputFileDir + "/" + str(self.index), "r")
        item = 'blank_item'
        while True:
            item = f.readline()
            if not item:
                break
            key = item.split(":0.")[0]
            value = float("0." + item.split(":0.")[1])
            self.attentionMap[key] = value
        f.close()

    def output_weight(self, output_filename):
        f = open(self.outputFileDir + "/" + output_filename, "w")
        for k, v in self.tokenMap.items():
            f.write(k + ":" + str(v['attention']) + '\n')
        f.close()


class Statement():
    def __init__(self, tokens, weight_file_dir="./weights", weight_file_name="latest_output", lang="java"):
        self.statements = []
        self.tokens = tokens
        self.statement_attention_map = []
        self.weight_file_dir = weight_file_dir
        self.weight_file_name = weight_file_name
        self.tokenIndexList = []
        self.lang = lang

    def merge_statements(self):
        if self.lang == 'java':
            start = 1
            end = 1
            result = []

            is_for_statement = False

            for i in range(1, len(self.tokens)):
                if self.tokens[i] == '</s>' and (i == len(self.tokens) - 1 or self.tokens[i+1] == '<s>'):
                    break
                if self.tokens[i] == '</s>':
                    start = i + 1
                    continue
                if is_for_statement:
                    if '{' in self.tokens[i]:
                        is_for_statement = False
                        end = i
                        result.append(self.tokens[start:end+1])
                        self.tokenIndexList.append([start, end])
                        start = end + 1
                    continue
                if self.tokens[i] == 'Ġfor' and self.tokens[i + 1] == 'Ġ(':
                    is_for_statement = True
                    start = i
                    continue
                if (self.tokens[i] == 'Ġ>' and self.tokens[i-1] == 'p' and self.tokens[i-2] == 'Ġ<') \
                        or ';' in self.tokens[i] or '{' in self.tokens[i] or '}' in self.tokens[i]:
                    end = i
                    result.append(self.tokens[start:end+1])
                    self.tokenIndexList.append([start, end])
                    start = end + 1
            self.statements = result
            return result, self.tokenIndexList
        elif self.lang == 'python':
            return self.merge_python_statements()

    def merge_python_statements(self):
        token_to_code_index = []
        start = self.tokens.index('</s>') + 1
        if self.tokens[start] == 'def':
            token_to_code_index.append([start, start])
            self.tokens[start] = 'Ġdef'
            start = start + 1
        index = start
        current_token_index = []
        while index < len(self.tokens):
            if self.tokens[index] == "</s>":
                if len(current_token_index) == 1:
                    current_token_index.append(index - 1)
                    token_to_code_index.append(current_token_index)
                break
            current_token = self.tokens[index]
            if current_token.startswith('Ġ'):
                if len(current_token_index) == 1:
                    current_token_index.append(index - 1)
                    token_to_code_index.append(current_token_index)
                current_token_index = [index]
            index += 1

        need_colon_keywords = ['if', 'else', 'class', 'elif', 'else', 'except', 'for', 'try', 'finally', 'while', 'def']
        start_keywords = ['assert', 'del', 'from', 'nonlocal', 'global', 'return', 'with', '#',
                          'raise']
        connect_keywords = ['and', '\'', '\"', 'as', 'False', 'True', 'in', '.', '(', '[', 'or', 'is', 'None', 'not',
                            '+', '-', '=', '*', '/', '%', '**', '//', '{', ',', '&']
        last_word_addition = [')', ']', '}']
        single_keywords = ['pass', 'break', 'continue']
        statements = []
        index = 0
        in_brace = 0
        start = 0

        while index < len(token_to_code_index):
            current_token = self.tokens[token_to_code_index[index][0]][1:]
            if current_token in ['{', '(', '[']:
                in_brace += 1
            if current_token in ['}', ')', ']']:
                in_brace -= 1
            if current_token == '\\':
                index += 1
                continue
            if current_token in need_colon_keywords:
                if current_token == 'for' and in_brace > 0:
                    index += 2
                    continue
                if start != index:
                    statements.append(self.tokens[token_to_code_index[start][0]:token_to_code_index[index][0]])
                    self.tokenIndexList.append([token_to_code_index[start][0], token_to_code_index[index-1][1]])
                start = index
                try:
                    end_keyword = self.tokens.index('Ġ:', token_to_code_index[start+1][0])
                except ValueError:
                    index += 1
                    continue
                except IndexError:
                    return statements, self.tokenIndexList
                statements.append(self.tokens[token_to_code_index[start][0]:end_keyword+1])
                self.tokenIndexList.append([token_to_code_index[start][0], end_keyword])
                for i in range(0, len(token_to_code_index)):
                    if token_to_code_index[i][0] <= end_keyword and token_to_code_index[i][1] >= end_keyword:
                        start = i + 1
                        index = i + 1
                        break
                continue
            # start statement
            if current_token in start_keywords + single_keywords:
                if start == index:
                    index += 1
                    continue
                statements.append(self.tokens[token_to_code_index[start][0]:token_to_code_index[index][0]])
                self.tokenIndexList.append([token_to_code_index[start][0], token_to_code_index[index-1][1]])
                index += 1
                start = index
                continue
            # connect statements
            prev_token = self.tokens[token_to_code_index[index-1][0]][1:]
            if prev_token not in connect_keywords + start_keywords and \
                    current_token not in connect_keywords + last_word_addition:
                if start == index:
                    index += 1
                    continue
                statements.append(self.tokens[token_to_code_index[start][0]:token_to_code_index[index][0]])
                self.tokenIndexList.append([token_to_code_index[start][0], token_to_code_index[index-1][1]])
                index += 1
                start = index
                continue
            index += 1
        if start < len(token_to_code_index):
            statements.append(self.tokens[token_to_code_index[start][0]:token_to_code_index[-1][1]])
            self.tokenIndexList.append([token_to_code_index[start][0], token_to_code_index[-1][1]])
        return statements, self.tokenIndexList

    def calculate_statement_weights(self):
        for statement in self.statements:
            total_weights = 0.0
            for token in statement:
                total_weights += self.attentionMap[token]
            total_weights /= len(statement)
            self.statement_attention_map.append({'statement': statement, 'attention': total_weights})
        return self.statement_attention_map

    def generate_attention_map(self):
        f = open(self.weight_file_dir + "/" + self.weight_file_name, "r")
        item = 'blank_item'
        while True:
            item = f.readline()
            if not item:
                break
            key = item.split(":0.")[0]
            value = float("0." + item.split(":0.")[1])
            self.attentionMap[key] = value
