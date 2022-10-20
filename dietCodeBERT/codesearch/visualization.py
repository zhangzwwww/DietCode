from matplotlib import pyplot as plt
import operator

try_statement_list = []
catch_statement_list = []
finally_statement_list = []
break_statement_list = []
continue_statement_list = []
return_statement_list = []
throw_statement_list = []
annotation_list = []
expression_list = []
while_statement_list = []
for_statement_list = []
if_statement_list = []
switch_statement_list = []
synchronized_statement_list = []
case_statement_list = []
method_declaration_statement_list = []
variable_declaration_statement_list = []
reassign_list = []
function_caller_list = []
setter_list = []
getter_list = []
logger_list = []


def average(list):
    return sum(list) / len(list)


def collect_data(attention_file_dir):
    try_statement_num = 0
    catch_statement_num = 0
    finally_statement_num = 0
    break_statement_num = 0
    continue_statement_num = 0
    return_statement_num = 0
    throw_statement_num = 0
    annotation_num = 0
    expression_num = 0
    while_statement_num = 0
    for_statement_num = 0
    if_statement_num = 0
    switch_statement_num = 0
    synchronized_statement_num = 0
    case_statement_num = 0
    method_declaration_statement_num = 0
    variable_declaration_statement_num = 0
    reassign_num = 0
    function_caller_num = 0
    setter_num = 0
    getter_num = 0
    logger_num = 0


    with open(attention_file_dir, 'r') as f:
        statement_item = 'blank_item'
        line_num = 1
        while True:
            statement_item = f.readline()
            if not statement_item:
                break
            statement = statement_item.split('  ||  ')[1]
            attention = float(statement_item.split('  ||  ')[0])
            line_num += 1
            if is_try_statement(statement):
                try_statement_num += 1
                try_statement_list.append(attention)
            elif is_catch_statement(statement):
                catch_statement_num += 1
                catch_statement_list.append(attention)
            elif is_finally_statement(statement):
                finally_statement_num += 1
                finally_statement_list.append(attention)
            elif is_break_statement(statement):
                break_statement_num += 1
                break_statement_list.append(attention)
            elif is_continue_statement(statement):
                continue_statement_num += 1
                continue_statement_list.append(attention)
            elif is_return_statement(statement):
                return_statement_num += 1
                return_statement_list.append(attention)
            elif is_throw_statement(statement):
                throw_statement_num += 1
                throw_statement_list.append(attention)
            elif is_annotation(statement):
                annotation_num += 1
                annotation_list.append(attention)
            elif is_while_statement(statement):
                while_statement_num += 1
                while_statement_list.append(attention)
            elif is_for_statement(statement):
                for_statement_num += 1
                for_statement_list.append(attention)
            elif is_if_statement(statement):
                if_statement_num += 1
                if_statement_list.append(attention)
            elif is_switch_statement(statement):
                switch_statement_num += 1
                switch_statement_list.append(attention)
            elif is_expression(statement):
                expression_num += 1
                expression_list.append(attention)
            elif is_synchronized_statement(statement):
                synchronized_statement_num += 1
                synchronized_statement_list.append(attention)
            elif is_case_statement(statement):
                case_statement_num += 1
                case_statement_list.append(attention)
            elif is_method_declaration_statement(statement):
                method_declaration_statement_num += 1
                method_declaration_statement_list.append(attention)
            elif is_variable_declaration_statement(statement):
                variable_declaration_statement_num += 1
                variable_declaration_statement_list.append(attention)
            # elif is_reassign_statement(statement):
                # reassign_num += 1
            #     reassign_list.append(attention)
            elif is_logger(statement):
                logger_num += 1
                logger_list.append(attention)
            elif is_setter(statement):
                setter_num += 1
                setter_list.append(attention)
            elif is_getter(statement):
                getter_num += 1
                getter_list.append(attention)
            elif is_function_caller(statement):
                function_caller_num += 1
                function_caller_list.append(attention)

    result = {'try': average(try_statement_list),
              'catch': average(catch_statement_list),
              'finally': average(finally_statement_list),
              'break': average(break_statement_list),
              'continue': average(continue_statement_list),
              'return': average(return_statement_list),
              'throw': average(throw_statement_list),
              'annotation': average(annotation_list),
              'expression': average(expression_list),
              'while': average(while_statement_list),
              'for': average(for_statement_list),
              'if': average(if_statement_list),
              'switch': average(switch_statement_list),
              'syncronized': average(synchronized_statement_list),
              'case': average(case_statement_list),
              'method': average(method_declaration_statement_list),
              'variable': average(variable_declaration_statement_list),
              # 'reas': average(reassign_list),
              'function': average(function_caller_list),
              'setter': average(setter_list),
              'getter': average(getter_list),
              'log': average(logger_list), }

    y = [result['try'], result['catch'], result['finally'], result['break'], result['continue'], result['return'],
         result['throw'], result['annotation'], result['expression'], result['while'], result['for'], result['if'],
         result['switch'], result['syncronized'], result['case'], result['method'], result['variable'],
         # result['reas'],
         result['function'], result['setter'], result['getter'], result['log']]
    x = [try_statement_num, catch_statement_num, finally_statement_num, break_statement_num, continue_statement_num,
         return_statement_num, throw_statement_num, annotation_num, expression_num, while_statement_num,
         for_statement_num, if_statement_num, switch_statement_num, synchronized_statement_num, case_statement_num,
         method_declaration_statement_num, variable_declaration_statement_num,
         # reassign_num,
         function_caller_num,
         setter_num, getter_num, logger_num]
    annotation = [a[0] for a in result.items()]

#     fig, ax = plt.subplots()
    # ax.scatter(x, y)
    # for i in range(len(x)):
        # plt.annotate(annotation[i], xy=(x[i],y[i]), xytext=(x[i]+1, y[i]))
#     plt.show()

    result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
    x = [a[0] for a in result]
    y = [a[1] for a in result]
    with open('./statement_classfication', 'w') as f:
        for a in result:
            f.write(str(a[0]) + '\n')
        for a in result:
            f.write(str(a[1]) + '\n')

    # plt.ylabel('Classification Rank')
    # plt.xlabel('Classification')
    # plt.bar(x, y, align='center')
    # plt.title('Java Statement Attention')
    # plt.show()

    return try_statement_list, catch_statement_list, finally_statement_list, break_statement_list,\
        continue_statement_list, return_statement_list, throw_statement_list, annotation_list, expression_list, \
        while_statement_list, for_statement_list, if_statement_list, switch_statement_list, \
        synchronized_statement_list, case_statement_list, method_declaration_statement_list, \
        variable_declaration_statement_list, reassign_list, function_caller_list


def is_logger(statement):
    if statement.startswith('log ') or statement.startswith('Logger ') or ' Log ' in statement \
            or '. print ' in statement or ' . println ' in statement or 'LOG ' in statement \
            or statement.startswith('logger ') or statement.startswith('debug .'):
        return True
    return False


def is_getter(statement):
    if '. get ' in statement:
        return True
    return False


def is_setter(statement):
    if '. set' in statement and ')' in statement:
        return True
    return False


def is_if_statement(statement):
    if statement.startswith('if (') or statement.startswith('else if ') or statement.startswith('else '):
        return True
    return False


def is_while_statement(statement):
    if statement.startswith('while'):
        return True
    return False


def is_synchronized_statement(statement):
    if statement.startswith('synchronized '):
        return True
    return False


def is_for_statement(statement):
    if statement.startswith('for ('):
        return True
    return False


def is_throw_statement(statement):
    if statement.startswith('throw'):
        return True
    return False


def is_method_declaration_statement(statement):
    if statement.startswith('public ') or statement.startswith('protected ') or statement.startswith('private ') \
            or statement.startswith('@ Over ride') or statement.startswith('@ Bench mark') \
            or statement.startswith('@ Gener ated ') or statement.startswith('@ Test') \
            or statement.endswith(') {\n') or ("throws " in statement and 'Exception' in statement):
        return True
    return False


def is_switch_statement(statement):
    if statement.startswith('switch'):
        return True
    return False


def is_return_statement(statement):
    if statement.startswith('return'):
        return True
    return False


def is_variable_declaration_statement(statement):
    if statement.startswith('String ') or statement.startswith('int ') or statement.startswith('float ') \
            or statement.startswith('boolean') or statement.startswith('long') or statement.startswith('List') \
            or statement.startswith('Array ') or 'new ' in statement or statement.startswith('Collection') \
            or statement.startswith('final ') or "= true" in statement or "= null" in statement \
            or statement.startswith('Object ') or "= \" string \"" in statement or statement.startswith('Map <') \
            or statement.startswith('Class <') \
            and '=' in statement:
        return True
    return False


def is_reassign_statement(statement):
    if ")" not in statement and "=" in statement or " = (" in statement:
        return True
    return False


def is_try_statement(statement):
    if statement.startswith('try'):
        return True
    return False


def is_catch_statement(statement):
    if statement.startswith('catch'):
        return True
    return False


def is_finally_statement(statement):
    if statement.startswith('finally'):
        return True
    return False


def is_break_statement(statement):
    if statement.startswith('break'):
        return True
    return False


def is_case_statement(statement):
    if statement.startswith('case'):
        return True
    return False


def is_continue_statement(statement):
    if statement.startswith('continue'):
        return True
    return False


def is_expression(statement):
    if " * = " in statement or "++ ;" in statement or "/\ = " in statement \
            or "+=" in statement or "-- ;" in statement or " / = " in statement:
        return True
    return False


def is_function_caller(statement):
    if statement.endswith(') ;\n') and '(' in statement:
        return True
    return False


def is_annotation(statement):
    if statement.startswith('//') or statement.endswith('< p >\n') or statement.startswith('*/') \
            or statement.startswith('/*'):
        return True
    return False


def reduce_statement(statement_list, range_list):
    range_list = sorted(range_list)
    start = 0
    end = 0
    result = []
    for r in range_list:
        start = end
        end = r
        sum = 0
        for line in statement_list:
            if line < end and line >= start:
                sum += 1
        result.append({'range': str(start) + '-' + str(end), 'number': sum})
    start = range_list[-1]
    sum = 0
    for line in statement_list:
        if line > start:
            sum += 1
    result.append({'range': str(start) + '-' + str(end), 'number': sum})
    return result


if __name__ == "__main__":
    try_statement_list, catch_statement_list, finally_statement_list, break_statement_list,\
        continue_statement_list, return_statement_list, throw_statement_list, annotation_list, expression_list, \
        while_statement_list, for_statement_list, if_statement_list, switch_statement_list, \
        synchronized_statement_list, case_statement_list, method_declaration_statement_list, \
        variable_declaration_statement_list, reassign_list, function_caller_list = collect_data(
            './output_statement/statement_attention')
