#!usr/bin/env python
#-*- coding: utf-8 -*-
## tools collected by zouxiaoyi

import sys
import os
import json
import re
import math

reload(sys)
sys.setdefaultencoding('utf8')

date_patterns =[re.compile('[1-2]{1}[0-9]{3}年度'),
                re.compile('[1-2]{1}[0-9]{3}年'),
                re.compile('[1-2]{1}[0-9]{3}'),
                re.compile('[0-9]{1,2}月[0-9]{1,2}日'),
                re.compile('[0-9]{1,2}月'),
                re.compile('[0-9]{1,2}日')]

delimiters='?!;？！。；…\n'
sentence_pat = re.compile(unicode('['+delimiters+']', 'utf8'))
min_split_pat = re.compile(unicode('[ 、,.:：，\(\)（）?!;？！。；…\n]', 'utf8'))

## code
def unicode_encode(input):
    if type(input) == str:
        return unicode(input, 'utf-8')
    elif type(input) == list:
        return [unicode(item, 'utf-8') for item in input]
    elif type(input) == tuple:
        return (unicode(item, 'utf-8') for item in input)
    elif type(input) == dict:
        new_dict = {}
        for k,v in input.items():
            new_dict[unicode(k, 'utf-8')] = unicode(v, 'utf-8')
        return new_dict
    else:
        if type(input) == unicode:
            return input
        sys.stderr.write('Not support input type for unicode_encode\n')
        sys.exit(1)

def utf8_encode(string):
    return string.decode('utf8','ignore').encode('utf8','ignore')

def gbk_utf8(string):
    return string.decode('gbk','ignore').encode('utf8','ignore')

def utf8_gbk(string):
    return string.decode('utf8','ignore').encode('gbk','ignore')

def to_utf8(string):
    utf8_string = ''
    try:
        utf8_string = string.decode('gb18030').encode('utf8')
    except:
        try:
            utf8_string = string.decode('gbk').encode('utf8')
        except:
            utf8_string = string
    return utf8_string

## string process
def remove_anchor_marker(input_str):
    re_str="<[^>]*>"
    pattern_com = re.compile(re_str)
    output_str = re.sub(pattern_com, '', input_str)
    return output_str

def remove_chinese_space(input_str):
    dummy = '\xe3\x80\x80'
    input_str = input_str.replace(dummy, '')
    if type(input_str) == unicode:
        dummy = unicode_encode(' ')
        input_str = input_str.replace(dummy, '')
    return input_str

def get_length_of_chinese_sentence(input_str):
    return len(unicode_encode(input_str))

def get_anchor_text(input_str):
    #anchor_re_rules = ["<a target=_blank href=(.*)>(.*?)</a", "<a.*data-lemmaid=\\\"(.*?)\\\".*>(.*?)</a",
    #                   "<a( href=\"#\"){0,1}>(.*?)</a", "<a.*>(.*?)</a>.*"]
    pat_anchor = re.compile('<a[^>]*>([^<]+)</a>')
    terms = pat_anchor.findall(input_str)
    return terms

def sentence_segmentation(content):
    if type(content) == type([]):
        content = '\n'.join(content)
    sentences = sentence_pat.split(content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences

def remove_given_successive_pos(pos_list, given_pos_list):
    flag_list = list()
    for pos in pos_list:
        fields = list(pos)
        if len(fields) != 2:
            continue
        flag_list.append(fields[1].strip())
    flag_list_str = ','.join(flag_list)
    for pos in given_pos_list:
        idx = flag_list_str.find(pos)
        if idx < 0:
            continue
        idx = len(flag_list_str[:idx].split(','))-1
        pos_len = len(pos.split(','))
        terms = list()
        for pos in pos_list[idx:idx+pos_len]:
            fields = list(pos)
            if len(fields) != 2:
                continue
            term = fields[0].strip()
            terms.append(term)
        if '代' in ''.join(terms):
            continue
        pos_list = pos_list[:idx] + pos_list[idx+pos_len:]
    return pos_list

def remove_given_pos(pos_list, given_pos_list):
    tag_list = []
    for pos in pos_list:
        fields = list(pos)
        if len(fields) != 2:
            continue
        term = fields[0].strip()
        flag = fields[1].strip()
        if flag in given_pos_list:
            continue
        if flag == 'j' and '大' in term:
            continue
        tag_list.append(term)
    tag = ''.join(tag_list)
    return tag

def remove_date_string(input_str):
    global date_patterns
    for p in date_patterns:
        input_str = re.sub(p, '', input_str)
    return input_str

def get_given_pos(pos_list, given_pos_list=['d', 'v', 'n']):
    noun_pos_list = ['an', 'Ng', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']
    if 'n' in given_pos_list:
        given_pos_list += noun_pos_list
    want_items = set()
    for pos in pos_list:
        fields = list(pos)
        if len(fields) != 2:
            continue
        if fields[1] in given_pos_list:
            want_items.add(fields[0].strip())
    return want_items

def get_given_successive_pos(pos_list, given_pos_list=['v,d,n', 'd,v,n', 'vn', 'n']):
    noun_pos_list = ['an', 'Ng', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']
    flag_list = list()
    item_list = list()
    for pos in pos_list:
        fields = list(pos)
        if len(fields) != 2:
            continue
        if fields[1].strip() in noun_pos_list:
            fields[1] = 'n'
        flag_list.append(fields[1].strip())
        item_list.append(fields[0].strip())
    flag_list_str = ','.join(flag_list)
    tags = ''
    for pos in given_pos_list:
        idx = flag_list_str.find(pos)
        if idx < 0:
            continue
        idx = len(flag_list_str[:idx].split(','))-1
        end_idx = idx + len(pos.split(','))
        while flag_list[end_idx] == 'n':
            end_idx += 1
            if end_idx == len(flag_list):break
        tag = ''.join(item_list[idx:end_idx])
        if tag not in tags:
            tags += tag + '#'
    return [tag.strip() for tag in tags.split('#') if len(tag.strip())>0]

def extract_tags_from_sentence(pos_list, stop_pos_list=['u','w','x','r'], stop_comb_pos_list=['m,q']):
    pos_list = remove_given_successive_pos(pos_list, stop_comb_pos_list)
    tag = remove_given_pos(pos_list, stop_pos_list)
    tag = remove_date_string(tag)
    return tag

## jieba NLP
def keyword_extract(sentence, method='tf-idf', topk=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')):
    if method == 'tf-idf':
        import jieba.analyse
        keywords = jieba.analyse.extract_tags(sentence, topk, withWeight, allowPOS)
    else:# method == 'textrank':
        import jieba
        keywords = jieba.analyse.textrank(sentence, topk, withWeight, allowPOS)
    keywords = [[val[0], round(val[1], 3)] for val in keywords]
    return keywords

def word_seg_pos(sentence):
    import jieba.posseg
    return list(jieba.posseg.cut(sentence))

## file
def load_char(input_file, is_unicode=False):
    my_set = set()
    with open(input_file) as hd:
        for line in hd:
            line = line.strip()
            if len(line) <= 0:
                continue
            if is_unicode == True:
                line = unicode_encode(line)
            my_set.add(line)
    return my_set

def get_fields(line, start_idx, end_idx, sep='\t'):
    fields = line.strip().split(sep)
    assert(len(fields) >= end_idx)
    return sep.join(fields[start_idx: end_idx])

def load_kv_file(input_file, key_idx, value_idx, is_unicode=False):
    info_dict = {}
    with open(input_file) as file_hd:
        for line in file_hd:
            fields = line.strip().split('\t')
            if len(fields) <= max(key_idx, value_idx):
                continue
            key = fields[key_idx]
            if is_unicode == True:
                #key = unicode_encode(fields[key_idx])
                key = fields[key_idx].decode('utf8', 'ignore')
            info_dict[key] = fields[value_idx]
    return info_dict

def load_baike_newid_kgid_dict(Reverse=False):
    info_file = '/home/disk2/zouxiaoyi/baike/data/baike_newid_kgid.dict'
    if Reverse == False:
        return load_kv_file(info_file, 0, 1)
    else:
        return load_kv_file(info_file, 1, 0)

# baike info
# 新ID 词条ID 义项ID kgID URL 词条名 词条描述（多义项） 所属分类 词条Tag
def load_baike_info(key_idx, value_idx):
    info_file = '/home/disk2/zouxiaoyi/baike/data/Baike_AllInfo.20150715'
    return load_kv_file(info_file, key_idx, value_idx)

# list
def add_two_list(list1, list2, weight):
    assert len(list1) == len(list2)
    new_list = [0] * len(list2)
    for i in range(len(list1)):
        new_list[i] = list1[i] + list2[i] * weight
    return new_list

def mean_std(input_list):
    mean = sum(input_list)/len(input_list)
    std = (sum([val**2 for val in input_list])/len(input_list) - mean**2)**0.5
    return mean, std

def threshold_using_gmm(data):
    from sklearn import mixture
    import numpy as np
    import copy
    data_cp = copy.deepcopy(data)
    data_cp.sort()
    X = np.vstack(data_cp)
    gmm = mixture.GMM(n_components=3, covariance_type='full')
    gmm.fit(X)
    predict = gmm.predict(X)
    if predict[0] == predict[-1]:
        return data_cp[0] - 1e-6
    threshold = 0.0
    for i in range(len(data)-1):
        if predict[i] != predict[i+1]:
            threshold = 0.5 * (data_cp[i] + data_cp[i+1])
            break
    return threshold

# distance
def jaccard_similarity(str1, str2):
    str1 = unicode_encode(str1)
    str2 = unicode_encode(str2)
    overlap_cnt = 0
    for word in str1:
        if word in str2:
            overlap_cnt += 1
    return float(overlap_cnt) / (len(str1)+len(str2)-overlap_cnt)

def cosine_similarity(vec1, vec2):
    if len(vec1)==0 or len(vec2)==0 or len(vec1) != len(vec2):
        return 0
    cross = 0.0
    norm_vec1 = 0.0
    norm_vec2 = 0.0
    for i in range(len(vec1)):
        cross += float(vec1[i]) * float(vec2[i])
        norm_vec1 += float(vec1[i]) * float(vec1[i])
        norm_vec2 += float(vec2[i]) * float(vec2[i])
    norm = (math.sqrt(norm_vec1) * math.sqrt(norm_vec2))
    cross = cross / norm if norm > 1e-6 else 0.0
    return round(cross, 4)

def levenshtein_distance(str1, str2):
    str1 = unicode_encode(str1)
    str2 = unicode_encode(str2)
    if len(str1) > len(str2):
        str1,str2 = str2,str1
    if len(str1) == 0:
        return len(str2)
    if len(str2) == 0:
        return len(str1)
    str1_length = len(str1) + 1
    str2_length = len(str2) + 1
    distance_matrix = [range(str2_length) for x in range(str1_length)] 
    for i in range(1, str1_length):
        for j in range(1, str2_length):
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]
            if str1[i-1] != str2[j-1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[str1_length-1][str2_length-1]

