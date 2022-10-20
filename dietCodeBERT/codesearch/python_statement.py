def is_try_statement(statement):
    if statement[0] == 'try':
        return True
    return False


def is_break_statement(statement):
    if statement[0] == 'break':
        return True
    return False


def is_finally_statement(statement):
    if statement[0] == 'finally':
        return True
    return False


def is_continue_statement(statement):
    if statement[0] == 'continue':
        return True
    return False


def is_return_statement(statement):
    if statement[0] == 'return':
        return True
    return False


def is_annotation(statement):
    if statement[0] == '#':
        return True
    return False


def is_while_statement(statement):
    if statement[0] == 'while':
        return True
    return False


def is_for_statement(statement):
    if statement[0] == 'for':
        return True
    return False


def is_if_statement(statement):
    if statement[0] == 'if':
        return True
    return False


def is_expression(statement):
    if '+' in statement or '-' in statement or '*' in statement or '/' in statement or '%' in statement or '+=' in statement or '-=' in statement \
            or '*=' in statement or '/=' in statement:
        return True
    return False


def is_method(statement):
    if statement[0] == 'def':
        return True
    return False


def is_variable(statement):
    if '=' in statement:
        return True
    return False


def is_function_caller(statement):
    if '(' in statement and statement[-1] == ')':
        return True
    return False
