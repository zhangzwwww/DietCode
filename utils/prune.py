import copy
import re
import visualization as st
import python_statement as ps

java_statement_classification_map = {}

python_statement_classification_map = {}

lowest_ranked_token = []


def get_token_attention():
    with open('./low_rated_word', 'r') as f:
        for token in f.readlines():
            lowest_ranked_token.append(token.replace('\n', ''))


get_token_attention()


def camel_case_split(str):
    RE_WORDS = re.compile(r'''
        [A-Z]+(?=[A-Z][a-z]) |
        [A-Z]?[a-z]+ |
        [A-Z]+ |
        \d+ |
        [^\u4e00-\u9fa5^a-z^A-Z^0-9]+
        ''', re.VERBOSE)
    return RE_WORDS.findall(str)

def underline(str):
    return str.split('_')

def merge_java_statements(code):
    statements = []
    tokens = code.split(' ')
    if tokens[0] == '@':
        if tokens[2] == '(':
            start = tokens.index(')')
            tokens = tokens[start+1:]
        else:
            tokens = tokens[2:]
    current_token = []
    for i in range(len(tokens)):
        token = tokens[i]
        token = camel_case_split(token)
        for t in token:
            current_token.append(t)
    tokens = current_token
    start = 0
    # try:
        # end_function_def = tokens.index('{')
        # statements.append(tokens[:end_function_def+1])
        # start = end_function_def+1
    # except ValueError:
        # pass
    index = start
    in_brace = 0
    endline_keyword = [';', '{', '}']
    while index < len(tokens):
        current_token = tokens[index]
        if current_token in ['(']:
            in_brace += 1
        elif current_token in [')']:
            in_brace -= 1
        if current_token in endline_keyword and in_brace > 0:
            index += 1
            continue
        if current_token in endline_keyword:
            statements.append(tokens[start:index+1])
            start = index + 1
            index += 1
            continue
        index += 1
    if start < len(tokens):
        statements.append(tokens[start:])
    return statements

def merge_python_statements(code):
    statements = []
    tokens = code.split(' ')
    current_token = []
    for token in tokens:
        for word in underline(token):
            current_token.append(word)
    tokens = current_token
    start = 0
    if tokens[0] == 'def':
        try:
            end_def = tokens.index(':')
            statements.append(tokens[:end_def+1])
            start = end_def+1
        except ValueError:
            pass
    index = start
    in_brace = 0
    need_colon_keywords = ['if', 'else', 'class', 'elif', 'else', 'except', 'for', 'try', 'finally', 'while', 'def']
    start_keywords = ['assert', 'del', 'from', 'nonlocal', 'global', 'return', 'with', '#',
                      'raise']
    connect_keywords = ['and', '\'', '\"', 'as', 'False', 'True', 'in', '.', '(', '[', 'or', 'is', 'None', 'not',
                            '+', '-', '=', '*', '/', '%', '**', '//', '{', ',', '&']
    last_word_addition = [')', ']', '}']
    single_keywords = ['pass', 'break', 'continue']
    while index < len(tokens):
        current_token = tokens[index]
        if current_token in ['{', '(', '[']:
            in_brace += 1
        if current_token in ['}' ,')', ']']:
            in_brace -= 1
        if tokens[index] == '\\':
            index += 1
            continue
        # ended with :
        if current_token in need_colon_keywords:
            if current_token == 'for' and in_brace > 0:
                index += 2
                continue
            if start != index:
                statements.append(tokens[start:index])
            start = index
            try:
                end_keyword = tokens.index(':', start+1)
            except ValueError:
                index += 1
                continue
            statements.append(tokens[start:end_keyword+1])
            start = end_keyword+1
            index = start
            continue
        # start statement
        if current_token in start_keywords + single_keywords:
            if start == index:
                index += 1
                continue
            statements.append(tokens[start:index])
            start = index
            index += 1
            continue
        # connect statements
        prev_token = tokens[index-1]
        if prev_token not in connect_keywords + start_keywords and \
                current_token not in connect_keywords + last_word_addition:
            if start == index:
                index += 1
                continue
            statements.append(tokens[start:index])
            start = index
            index += 1
            continue
        index += 1
    statements.append(tokens[start:])
    result = []
    for statement in statements:
        if len(statement) == 1 and statement[0] not in need_colon_keywords + start_keywords + connect_keywords + \
                last_word_addition + single_keywords:
            if len(result) > 0:
                result[-1].append(statement[0])
            else:
                result.append(statement)
        else:
            result.append(statement)

    return result


def get_java_statement_classification(statement):
    if st.is_try_statement(statement):
        return 'try', java_statement_classification_map['try']
    elif st.is_catch_statement(statement):
        return 'catch', java_statement_classification_map['catch']
    elif st.is_finally_statement(statement):
        return 'finally', java_statement_classification_map['finally']
    elif st.is_break_statement(statement):
        return 'break', java_statement_classification_map['break']
    elif st.is_continue_statement(statement):
        return 'continue', java_statement_classification_map['continue']
    elif st.is_return_statement(statement):
        return 'return', java_statement_classification_map['return']
    elif st.is_throw_statement(statement):
        return 'throw', java_statement_classification_map['throw']
    elif st.is_annotation(statement):
        return 'annotation', java_statement_classification_map['annotation']
    elif st.is_while_statement(statement):
        return 'while', java_statement_classification_map['while']
    elif st.is_for_statement(statement):
        return 'for', java_statement_classification_map['for']
    elif st.is_if_statement(statement):
        return 'if', java_statement_classification_map['if']
    elif st.is_switch_statement(statement):
        return 'switch', java_statement_classification_map['switch']
    elif st.is_expression(statement):
        return 'expression', java_statement_classification_map['expression']
    elif st.is_synchronized_statement(statement):
        return 'synchronized', java_statement_classification_map['synchronized']
    elif st.is_case_statement(statement):
        return 'case', java_statement_classification_map['case']
    elif st.is_method_declaration_statement(statement):
        return 'method', java_statement_classification_map['method']
    elif st.is_variable_declaration_statement(statement):
        return 'variable', java_statement_classification_map['variable']
    elif st.is_logger(statement):
        return 'logger', java_statement_classification_map['logger']
    elif st.is_setter(statement):
        return 'setter', java_statement_classification_map['setter']
    elif st.is_getter(statement):
        return 'getter', java_statement_classification_map['getter']
    elif st.is_function_caller(statement):
        return 'function', java_statement_classification_map['function']
    return 'None', 0.0001


def get_python_statement_classification(statement):
    if ps.is_try_statement(statement):
        return 'try',  python_statement_classification_map['try']
    elif ps.is_break_statement(statement):
        return 'break', python_statement_classification_map['break']
    elif ps.is_finally_statement(statement):
        return 'finally', python_statement_classification_map['finally']
    elif ps.is_continue_statement(statement):
        return 'continue', python_statement_classification_map['continue']
    elif ps.is_return_statement(statement):
        return 'return', python_statement_classification_map['return']
    elif ps.is_annotation(statement):
        return 'annotation', python_statement_classification_map['annotation']
    elif ps.is_while_statement(statement):
        return 'while', python_statement_classification_map['while']
    elif ps.is_for_statement(statement):
        return 'for', python_statement_classification_map['for']
    elif ps.is_if_statement(statement):
        return 'if', python_statement_classification_map['if']
    elif ps.is_expression(statement):
        return 'expression', python_statement_classification_map['expression']
    elif ps.is_method(statement):
        return 'method', python_statement_classification_map['method']
    elif ps.is_variable(statement):
        return 'variable', python_statement_classification_map['variable']
    elif ps.is_function_caller(statement):
        return 'function', python_statement_classification_map['function']
    return 'None', 0.0001


class Code_Reduction():  # self.statement_attention: statement categories' attention. Form as:
    #      [{category: 'if statement', content: 'statement content', attention: 0.01, length: 10}]
    # self.token_attention: token attention. Form as {'a': 0.01, 'b': 0.02}
    def __init__(self, code, lang='java', targetLength=100, **kwargs):
        self.code = code
        self.lang = lang
        self.targetLength = targetLength
        self.result = []
        self.generate_statements()

    def generate_statements(self):
        statements = None
        self.statements = []
        if self.lang == 'python':
            statements = merge_python_statements(self.code)
            self.statements = []
            for statement in statements:
                category, attention = get_python_statement_classification(statement)
                current_statement = {'category': category, 'content': statement,
                                     'length': len(statement), 'attention': attention}
                self.statements.append(current_statement)
        elif self.lang == 'java':
            statements = merge_java_statements(self.code)
            self.statements = []
            for statement in statements:
                category, attention = get_java_statement_classification(' '.join(statement))
                current_statement = {'category': category, 'content': statement,
                                     'length': len(statement), 'attention': attention}
                self.statements.append(current_statement)

    def generate_statement_attention(self, attention_file_dir='./statement_attention'):
        self.statement_attention = []
        # self.statements = self.statements(key=lambda item: item['attention'], reverse=True)

    def generate_token_attention(self, attention_file_dir='./token_attention'):
        self.token_attention = {}

    def get_statement_attention(self, statement):
        pass

    def prune_lowest_ranked_token(self, statements, prune_num):
        result = []
        # check pruning items
        candidate = []
        for statement in statements:
            for token in statement:
                if token in lowest_ranked_token:
                    attention_pos = lowest_ranked_token.index(token)
                    if len(candidate) <= prune_num:
                        candidate.append(attention_pos)
                    elif attention_pos < max(candidate) and attention_pos not in candidate:
                        candidate.remove(max(candidate))
                        candidate.append(attention_pos)
        # prune phase
        pruned_num = 0
        need_check = True
        candidate = [lowest_ranked_token[x] for x in candidate]
        for statement in statements:
            if not need_check:
                result.append(statement)
                continue
            current_statement = []
            for token in statement:
                if token in candidate and need_check:
                    pruned_num += 1
                    if pruned_num >= prune_num:
                        need_check = False
                else:
                    current_statement.append(token)
                continue
            result.append(current_statement)
        return result

    def zero_one_backpack(self):
        # after the 0-1 backpack problem solution get the chosen statements we prefer to reduce tokens insteam of add
        # tokens so we choose to increase the target length by the max length of all the statements so that the solution
        # will at least consist of more than one statement than the solution of the previous target length.
        max_length = 0
        for statement in self.statements:
            if statement['length'] > max_length:
                max_length = statement['length']
        max_length += self.targetLength
        dp = [[{'attention': 0.0, 'statements': []}
               for i in range(max_length + 1)] for j in range(len(self.statements) + 1)]
        for i in range(1, len(self.statements) + 1):
            for j in range(1, max_length + 1):
                current_map = {'attention': dp[i-1][j]['attention'],
                               'statements': copy.deepcopy(dp[i-1][j]['statements'])}
                dp[i][j] = current_map
                if j >= self.statements[i-1]['length']:
                    if dp[i][j]['attention'] < \
                            dp[i-1][j-self.statements[i-1]['length']]['attention'] + self.statements[i-1]['attention']:
                        dp[i][j]['attention'] = dp[i-1][j-self.statements[i-1]
                                                        ['length']]['attention'] + self.statements[i-1]['attention']
                        dp[i][j]['statements'] = copy.deepcopy(
                            dp[i-1][j-self.statements[i-1]['length']]['statements'])
                        dp[i][j]['statements'].append(i-1)
        return dp[-1][-1]['attention'], dp[-1][-1]['statements']

    def prune(self, **kwargs):
        # adding statments: greedy
        total_attention, chosen_statements = self.zero_one_backpack()
        current_length = 0
        result = []
        for statement_index in chosen_statements:
            current_length += self.statements[statement_index]['length']
            result.append(self.statements[statement_index]['content'])
        pruned_token_num = current_length - self.targetLength
        if pruned_token_num > 0:
            result = self.prune_lowest_ranked_token(result, pruned_token_num)
        return ' '.join(' '.join(x) for x in result)
