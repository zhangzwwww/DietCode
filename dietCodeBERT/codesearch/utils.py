# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

from weights import (WeightOutputer, Statement)
from prune import Code_Reduction

import numpy as np
import random
import re
import csv
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)
outputFileIndex = 1
low_rated_tokens = []

origin_code_length = 0.0
pruned_code_length = 0.0


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines


class CodesearchProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if (set_type == 'test'):
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, prune_strategy='None', lang='java'):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    global origin_code_length
    global pruned_code_length
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)[:50]

        tokens_b = None
        if example.text_b:
            # TODO: replace string and integer with identical content
            origin_code_length += len(example.text_b)
            if lang == 'java':
                example.text_b = assimilate_code_string_and_integer(example.text_b)
            example.text_b = delete_code_pattern(example.text_b, prune_strategy, lang)
            
            pruned_code_length += len(example.text_b)

            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    with open('./pruned-rate', 'w') as f:
        f.write(str(origin_code_length) + '\n')
        f.write(str(pruned_code_length) + '\n')
        f.write(str(pruned_code_length / origin_code_length))
    return features


def generate_low_rated_tokens(token_file_dir='./', pruned_rate=0.5):
    low_rated_tokens = []
    with open(token_file_dir, 'r') as f:
        token = 'blank_item'
        while True:
            token = f.readline()
            if not token:
                break
            low_rated_tokens.append(token.replace('\n', ''))
    return low_rated_tokens


def delete_code_with_variable_declaration(code, lang='java'):
    if lang == 'java':
        start, end = 0, 0
        statements = []
        modifiers = ['final', 'static']
        for i in range(len(code)):
            if code[i] == '}' or code[i] == '{' or code[i] == ';':
                end = i
                statements.append(code[start: end+1].strip())
                start = end + 1
        for i in range(len(statements)):
            tokens = statements[i].split(' ')
            if '=' not in tokens:
                continue
            equationMarkIndex = tokens.index('=')
            if equationMarkIndex < 2:
                if tokens[equationMarkIndex + 1] == 'new':
                    statements[i] == ''
                continue
            definitionPart = tokens[:equationMarkIndex+1]
            if definitionPart[-2] == '>' and '<' in definitionPart:
                statements[i] = ''
                continue
            for index in range(len(tokens)):
                if tokens[index] not in modifiers and index + 2 < len(tokens) and tokens[index+2] == '=':
                    statements[i] = ''
                    break
        return ' '.join(statements)


def delete_code_pattern(code, strategy='None', lang='java', **kwargs):
    result = ''
    if strategy == 'method-return':
        if lang == 'java':
            main_body_start = code.find('{') + 1
            result = code[:main_body_start]
            index = main_body_start
            while index < len(code):
                i = code.find(' return ', index)
                index = i + 1
                if i == -1:
                    result = result + " }"
                    break
                result += code[i:code.find(';', i) + 1]
        elif lang == 'python':
            statements = merge_python_statements(code)
            result = ' '
            for statement in statements:
                if len(statement) == 0:
                    continue
                if statement[0] == 'def' or statement[0] == 'return':
                    result = result + ' '.join(statement) + ' '
    elif strategy == 'trim':
        rate = 0.5
        # result = code[:int(len(code)*rate)]
        result = prune_tokens(code, 0.5)
        return code[:int(len(code)*rate)]
    elif strategy == 'slim':
        if lang == 'java':
            length = 80
            if 'length' in kwargs.keys():
                length = int(kwargs.keys())
            reduction = Code_Reduction(code, lang=lang, targetLength=length)
            result = reduction.prune()
        elif lang == 'python':
            length = 120
            reduction = Code_Reduction(code, lang=lang, targetLength=length)
            result = reduction.prune()
    elif strategy == 'variable':
        if lang == 'java':
            result = delete_code_with_variable_declaration(code)
        elif lang == 'python':
            pass
    elif strategy == 'loop':
        if lang == 'java':
            pass
        elif lang == 'python':
            statements = merge_python_statements(code)
            result = ' '
            for statement in statements:
                if len(statement) == 0:
                    continue
                if statement[0] != 'if' and statement[0] != 'while' and statement[0] != 'for':
                    result = result + ' '.join(statement) + ' '
    elif strategy == 'token':
        global low_rated_tokens
        if len(low_rated_tokens) == 0:
            logger.info("generate low rated tokens...")
            low_rated_tokens = generate_low_rated_tokens('./low_rated_word')
        if lang == 'java':
            result = ' '.join(prune_tokens(code, low_rated_tokens))
        elif lang == 'python':
            result = ' '.join(prune_tokens(code, low_rated_tokens))
    elif strategy == 'random':
        rate = 0.7
        if 'rate' in kwargs.keys():
            rate = kwargs['rate']
        result = random_prune_code_with_ratio(code, rate)
    elif strategy == 'None':
        return code
    else:
        return code
    return result


def assimilate_code_string_and_integer(code, string_mask=" string ", number_mask="10"):
    quotation_index_list = []
    for i in range(0, len(code)):
        if code[i] == "\"":
            quotation_index_list.append(i)
    for i in range(len(quotation_index_list) - 1, 0, -2):
        code = code[:quotation_index_list[i-1] + 1] + string_mask + code[quotation_index_list[i]:]

    tokens = code.split(" ")
    for i in range(0, len(tokens)):
        if is_number(tokens[i]):
            tokens[i] = number_mask
    code = " ".join(tokens)

    return code


def random_prune_code_with_ratio(code, rate=0.3):
    def random_select_tokens(tokens, ratio):
        pruned_index = sorted(random.sample(range(0, len(tokens)), int(len(tokens) * rate)), reverse=True)
        for index in pruned_index:
            del tokens[index:index+1]
        return tokens
    tokens = code.split(' ')
    result = random_select_tokens(tokens, rate)
    return ' '.join(result)


def prune_tokens(code, rate, low_rated_tokens = []):
    def camel_case_split(str):
        RE_WORDS = re.compile(r'''
        [A-Z]+(?=[A-Z][a-z]) |
        [A-Z]?[a-z]+ |
        [A-Z]+ |
        \d+ |
        [^\u4e00-\u9fa5^a-z^A-Z^0-9]+
        ''', re.VERBOSE)
        return RE_WORDS.findall(str)

    def is_not_low_rated_token(token):
        if token not in low_rated_tokens:
            return token
        return ''
    tokens = code.split(' ')
    result = []
    for token in tokens:
        camel_case_words = code.split('_')
        # low_rated_words = list(map(is_not_low_rated_token, camel_case_words))
        result.append(''.join(camel_case_words))
    result = result[:int(len(result)*rate)]
    return result


def merge_java_statements(code):
    statements = []
    tokens = code.split(' ')
    start = 0
    try:
        end_function_def = tokens.index('{')
        statements.append(tokens[:end_function_def+1])
        start = end_function_def+1
    except ValueError:
        pass
    index = start
    in_brace = 0
    endline_keyword = [';', '{']
    while index < len(tokens):
        current_token = tokens[index]
        if current_token in ['(']:
            in_brace += 1
        elif current_token in [')']:
            in_brace -= 1
        if current_token == ';' and in_brace > 0:
            index += 1
            continue
        if current_token in endline_keyword:
            statements.append(tokens[start:index+1])
            start = index + 1
            index += 1
    if start < len(tokens):
        statements.append(tokens[start:])
    return statements


def merge_python_statements(code):
    statements = []
    tokens = code.split(' ')
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


def output_weights(attentions, tokens, output_dir="./weights", output_statement=True, output_words=True, lang="java"):
    wo = WeightOutputer()
    wo.set_output_file_dir(output_dir)
    global outputFileIndex
    for j in range(0, len(attentions[0])):
        statementGenerator = Statement(tokens[j], lang='python')
        statements, tokenIndexList = statementGenerator.merge_statements()
        for i in range(len(attentions)-1, len(attentions)):
            if output_statement:
                output_layer_attention(i, attentions[i][j], tokenIndexList, "./output/" + str(outputFileIndex))
        if output_statement:
            output_tokens("./output/" + str(outputFileIndex), tokens[j])

        if output_words:
            if outputFileIndex % 100 == 50:
                logger.info("start output token weights into file" + str(outputFileIndex))
                wo.output_weight(str(outputFileIndex))
                logger.info("finish output token weights")
        outputFileIndex += 1


def output_layer_attention(layer, attentions, tokenIndexList, output_file_dir):
    layer_attention_list = []
    for statement_range in tokenIndexList:
        statement_start = statement_range[0]
        statement_end = statement_range[1]
        current_attention_list = [0.0] * len(attentions[0][0])
        for head in attentions:
            for i in range(0, len(head)):
                for j in range(statement_start, statement_end + 1):
                    current_attention_list[j] += head[i][j]
        for i in range(0, len(current_attention_list)):
            current_attention_list[i] = current_attention_list[i] / len(attentions[0][0]) / len(attentions)
        layer_attention_list.append(current_attention_list)
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    with open(output_file_dir + '/layer_' + str(layer), 'w') as f:
        for statement in layer_attention_list:
            f.write(str(statement) + '\n')
    return layer_attention_list


def output_tokens(output_file_dir, tokens):
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    with open(output_file_dir + "/tokens", 'w') as f:
        for token in tokens:
            f.write("%s\n" % token)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "codesearch":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "codesearch": CodesearchProcessor,
}

output_modes = {
    "codesearch": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "codesearch": 2,
}
